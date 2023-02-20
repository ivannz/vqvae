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
from vq.utils import VQEMAUpdater, VQLossHelper

# the arch and embedding settings were borrowed from
#    https://keras.io/examples/generative/vq_vae/
config = {
    "n_embeddings": 64,
    "n_latents": 16,
    "f_vq_alpha": 0.25,
    "f_ema_update": True,
    "f_lr": 1e-3,
    "n_batch_size": 128,
    "n_epochs": 5,
}

wandb.init(
    project="vq-vae",
    save_code=False,
    tags=["mnist"],
    mode="online",
    config=config,
)

cfg = wandb.config

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
feeds = {k: DataLoader(datasets[src], **spec)
         for k, (src, spec) in feeds_specs.items()}

# get the reference batches
refx, refy = next(iter(feeds["reference"]))

# build the vqvae
model = nn.Sequential(  # FIXME 28x28 -->> 29x29
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


# run the vqvae in inference mode
@torch.inference_mode()
def apply(model: nn.Module, X: Tensor) -> tuple[Tensor, Tensor]:
    vqout = model[:6](X)
    recx = model[6:](vqout)[..., :28, :28]
    return recx.clamp(-0.5, +0.5), vqout.indices


optim = torch.optim.AdamW(model.parameters(), lr=cfg.f_lr)

# if update is False then we do not make ema updates, but use
#  it to compute diagnostic entropy (health of the clustering)
#  alpha is the EMA decay rate
# XXX 1/4 seems to work best in this example, 0.5 and higher --
#  too fast affinity switches, lower than 0.1 -- clustering is
#  too suboptimal and slow.
# XXX centroid which try to capture the same clusters seem to be
#  too unstable and jittery. Could we use this to prune quantization?
#  On the otther hand, we could try adding new centroids to decrease
#  cluster overall clsuter variance (cluster splitting).
hlp = VQLossHelper(model, reduction="mean")
ema = VQEMAUpdater(model, alpha=cfg.f_vq_alpha, update=cfg.f_ema_update)

n_display_every = 25
n_display, n_updates = 0, 0
for ep in range(cfg.n_epochs):
    model.train()
    for bx, by in tqdm.tqdm(feeds["train"], ncols=50, disable=False):
        # use the loss helper and the EMA centroid updates
        with hlp, ema:
            out = model(bx)[..., :28, :28]

        # the AE loss is the simple reconstruction loss, while the VQ losses
        #  are cluster-commitment and the embedding loss
        ae_loss = F.mse_loss(out, bx, reduction="mean")
        vq_ell = sum(hlp.finish().values())

        # backprop
        loss = ae_loss + vq_ell
        optim.zero_grad(True)
        loss.backward()
        optim.step()

        # this does EMA updates, if they are enabled, otherwise has no effect
        ema.step()

        n_updates += 1

        # display reconstriction success
        if n_updates >= n_display:
            n_display = n_updates + n_display_every

            model.eval()
            outx, outc = apply(model, refx)
            image = wandb.Image(make_grid(outx.movedim(1, -1), aspect=(1, 1)))
            wandb.log({"reconstruction": image}, commit=False)
            model.train()

        # log some history
        wandb.log({
            **{"entropy/" + k: v for k, v in ema.named_entropy.items()},
            "loss/vq": float(vq_ell),
            "loss/ae": float(ae_loss),
        }, commit=True)
