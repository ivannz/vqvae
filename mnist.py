import tqdm
import torch
from matplotlib import pyplot as plt

from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader

from torchvision.utils import make_grid

# VQ is basically an online non-stationary k-means, and thus can
#  be considered an unsupervised denoiser
from vq import VectorQuantizedVAE, ExtractEmbeddings
from vq.utils import VQEMAUpdater, VQLossHelper


# get the MINST dataset
transform = Compose([ToTensor(), Normalize((0.5,), (1.0,))])
datasets = {
    "train": MNIST("./", train=True, transform=transform, download=True),
    "test": MNIST("./", train=False, transform=transform, download=True),
}

# split and prep the data feeds
feeds_specs = {
    "train": ("train", {"batch_size": 128, "shuffle": True}),
    "test": ("test", {"batch_size": 128, "shuffle": False}),
    "reference": ("train", {"batch_size": 64, "shuffle": True})
}
feeds = {k: DataLoader(datasets[src], **spec)
         for k, (src, spec) in feeds_specs.items()}

# get the reference batches
refx, refy = next(iter(feeds["reference"]))

# build the vqvae
n_embeddings, n_latents = 64, 16
model = nn.Sequential(  # FIXME 28x28 -->> 29x29
    nn.Conv2d(1, 32, 3, 2),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, 2),
    nn.ReLU(),
    nn.Conv2d(64, n_latents, 1, 1),

    VectorQuantizedVAE(n_embeddings, n_latents, -3),

    ExtractEmbeddings(),
    nn.ConvTranspose2d(n_latents, 64, 3, 2),
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


optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

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
ema = VQEMAUpdater(model, alpha=0.25, update=True)

hist, n_display, n_updates, n_display_every = [], 0, 0, 25
for ep in range(15):
    for bx, by in tqdm.tqdm(feeds["train"], ncols=50, disable=False):
        model.train()

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

        # log some history
        hist.append((*ema.entropy.values(), float(vq_ell), float(ae_loss)))

        if n_updates >= n_display:
            n_display = n_updates + n_display_every
            plt.close()
            plt.show()
            plt.ion()

            model.eval()
            fig, axes = plt.subplots(2, 2, dpi=80, figsize=(7, 7))
            (ax1, ax2), (ax3, ax4) = axes
            outx, outc = apply(model, refx)
            ax1.imshow(make_grid(refx).movedim(0, -1) + 0.5)
            ax2.imshow(make_grid(outx).movedim(0, -1) + 0.5)

            ent, vq, ae = zip(*hist)
            ax3.semilogy(ae, label="ae")
            ax3.semilogy(vq, label="vq")
            ax3.legend()

            ax4.plot(ent)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)

            model.train()

    model.eval()
