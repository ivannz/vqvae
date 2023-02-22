import os
import git
import time
import tqdm
import torch  # SEGFAULTS unless first!
# from matplotlib import pyplot as plt

import wandb
# import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader

from grid import make_grid

# VQ is basically an online non-stationary k-means, and thus can
#  be considered an unsupervised denoiser
from vq import VectorQuantizedVAE, ExtractEmbeddings

from argparse import Namespace


# PixeCNN impl taken from
#   https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html
class MaskedConv2d(nn.Conv2d):
    """Convolution with leeft-to-right top-to-bottom causal masking."""
    def __init__(self, kind: str = ("b", "f"), *args, **kwargs) -> None:
        assert kind in ("b", "f")
        super().__init__(*args, **kwargs)

        # setup the mask
        mask = torch.ones(self.kernel_size)
        j = mask.numel() // 2
        mask.ravel()[j + (1 if kind == "b" else 0):] = 0
        # XXX what is so special about the left-to-right top-to-bottom mask?
        self.register_buffer("mask", mask)  # torch.ones(self.kernel_size).triu_()

    def forward(self, input: Tensor) -> Tensor:
        return super()._conv_forward(input, self.weight * self.mask, self.bias)


class MaskedConvolution(nn.Module):

    def __init__(self, c_in, c_out, mask, **kwargs):
        """
        Implements a convolution with mask applied on its weights.
        Inputs:
            c_in - Number of input channels
            c_out - Number of output channels
            mask - Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
                   the convolution should be masked, and 1s otherwise.
            kwargs - Additional arguments for the convolution
        """
        super().__init__()
        # For simplicity: calculate padding automatically
        kernel_size = (mask.shape[0], mask.shape[1])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation*(kernel_size[i]-1)//2 for i in range(2)])
        # Actual convolution
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, **kwargs)

        # Mask as buffer => it is no parameter but still a tensor of the module
        # (must be moved with the devices)
        self.register_buffer('mask', mask[None,None])

    def forward(self, input: Tensor) -> Tensor:
        return self.conv._conv_forward(input, self.conv.weight * self.mask, self.conv.bias)


class VerticalStackConvolution(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size//2+1:,:] = 0

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[kernel_size//2,:] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


class HorizontalStackConvolution(MaskedConvolution):

    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(1,kernel_size)
        mask[0,kernel_size//2+1:] = 0

        # For the very first convolution, we will also mask the center pixel
        if mask_center:
            mask[0,kernel_size//2] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


class GatedMaskedConv(nn.Module):

    def __init__(self, c_in, **kwargs):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        self.conv_vert = VerticalStackConvolution(c_in, c_out=2*c_in, **kwargs)
        self.conv_horiz = HorizontalStackConvolution(c_in, c_out=2*c_in, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(2*c_in, 2*c_in, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = nn.Conv2d(c_in, c_in, kernel_size=1, padding=0)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out


class PixelCNN(nn.Module):

    def __init__(self, c_in: int = 64, c_hidden: int = 128):
        super().__init__()
        self.c_in, self.c_hidden = c_in, c_hidden

        # Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(c_in, c_hidden, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(c_in, c_hidden, mask_center=True)
        # Convolution block of PixelCNN. We use dilation instead of downscaling
        self.conv_layers = nn.ModuleList([
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=2),
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=4),
            GatedMaskedConv(c_hidden),
            GatedMaskedConv(c_hidden, dilation=2),
            GatedMaskedConv(c_hidden)
        ])
        # Output classification convolution (1x1)
        self.conv_out = nn.Conv2d(c_hidden, c_in, kernel_size=1, padding=0)

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor with integer values between 0 and 255.
        """
        x = F.one_hot(input, self.c_in).movedim(-1, -3).float()

        # Initial convolutions
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        # Gated Convolutions
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        # 1x1 classification convolution
        # Apply ELU before 1x1 convolution for non-linearity on residual connection
        return self.conv_out(F.elu(h_stack))


@torch.inference_mode()
def sample(model: PixelCNN, shape: tuple[int]) -> Tensor:
    """Very slow pix-by-pix generation"""
    B, H, W = shape

    canvas = torch.full((B, H, W), 0, dtype=int)
    for h in range(H):
        for w in range(W):
            # if (canvas[:, h, w] != -1).all():
            #     continue

            # predict the central code
            logits = model(canvas[:, :h + 1, :])[..., h, w]
            probs = logits.softmax(dim=-1)
            canvas[:, h, w] = probs.multinomial(1).squeeze(-1)

    return canvas


# load the trained vq
ckpt = Namespace(**torch.load("./checkpoints/ckpt-20230222_003848-vtjw37qr.pt"))
cfg = Namespace(**ckpt.config)

# build the vqvae (copied from mnist.py)
vqvae = nn.Sequential(  # FIXME 28x28 -->> 29x29
    nn.Conv2d(1, 32, 3, 2),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, 2),
    nn.ReLU(),
    nn.Conv2d(64, cfg.n_latents, 1, 1),

    VectorQuantizedVAE(cfg.n_embeddings, cfg.n_latents, -3),

    ExtractEmbeddings(),
    nn.ConvTranspose2d(cfg.n_latents, 64, 3, 2),
    nn.ReLU(),
    nn.ConvTranspose2d(64, 32, 3, 2),
    nn.ReLU(),
    nn.ConvTranspose2d(32, 1, 3, 1),
)
vqvae.load_state_dict(ckpt.model)


@torch.no_grad()
def apply(model: nn.Module, X: Tensor) -> tuple[Tensor, Tensor]:
    """run the vqvae in inference mode"""
    vqout = model[:6](X)
    recx = model[6:](vqout)[..., :28, :28]
    return recx.clamp(-0.5, +0.5), vqout.indices


wandb.init(
    project="vq-vae-pixelcnn",
    save_code=False,
    tags=["mnist"],
    mode="online",
    config=dict(
        vavae=vars(cfg),
    ),
)

# get the MINST dataset
transform = Compose([ToTensor(), Normalize((0.5,), (1.0,))])
datasets = {
    "train": MNIST("./", train=True, transform=transform, download=True),
    "test": MNIST("./", train=False, transform=transform, download=True),
}

# split and prep the data feeds
gen = torch.Generator()
gen.manual_seed(0xdeadc0de)  # for fixed random sample of reference images

feeds_specs = {
    "train": ("train", {"batch_size": cfg.n_batch_size, "shuffle": True}),
    "test": ("test", {"batch_size": 128, "shuffle": False}),
    "reference": ("train", {"batch_size": 64, "shuffle": True, "generator": gen})
}
feeds = {k: DataLoader(datasets[src], **spec) for k, (src, spec) in feeds_specs.items()}

# fetch reference batch's codes
vqvae.eval()
refx, refy = next(iter(feeds["reference"]))
_, refc = apply(vqvae, refx)


# train the code generator model
model = PixelCNN(cfg.n_embeddings, 128)
optim = torch.optim.AdamW(model.parameters(), lr=cfg.f_lr)


@torch.inference_mode()
def generate(shape: tuple[int]) -> None:
    codes = sample(model, shape)
    return vqvae[5:](codes)[..., :28, :28].clamp(-0.5, +0.5)


for ep in range(30):
    # generate
    model.eval()
    recx = generate(refc.shape).movedim(1, -1)
    image = wandb.Image(make_grid(recx, aspect=(1, 1)))
    wandb.log({"generation": image}, commit=False)

    model.train()
    for bx, by in tqdm.tqdm(feeds["train"], ncols=50, disable=False):
        # fetch the learnt embedding codes
        _, outc = apply(vqvae, bx)

        nll = F.cross_entropy(model(outc), outc, reduction="mean")

        # backprop
        optim.zero_grad(True)
        nll.backward()
        optim.step()

        # get the accuracy on the ref sample
        with torch.inference_mode():
            pred = model(refc)

        wandb.log({
            "error": float((pred.argmax(1) != refc).float().mean()),
            "loss": float(nll),
        }, commit=True)

# generate
model.eval()
recx = generate(refc.shape).movedim(1, -1)
image = wandb.Image(make_grid(recx, aspect=(1, 1)))
wandb.log({"generation": image}, commit=True)

# save the trained model
__dttm__ = time.strftime("%Y%m%d_%H%M%S")

ckpt = os.path.abspath('./checkpoints')

os.makedirs(ckpt, exist_ok=True)
torch.save(
    dict(
        __dttm__=__dttm__,
        __commit__=git.Repo(".").head.commit.hexsha,
        config=wandb.config.as_dict(),
        model=model.state_dict(),
    ), os.path.join(ckpt, f"pix-ckpt-{__dttm__}-{wandb.run.id}.pt")
)
