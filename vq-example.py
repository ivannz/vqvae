import tqdm
import torch
import numpy as np

from torch import nn, Tensor
from torch.nn import functional as F

from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


# VQ is basically an online non-stationary k-means, and thus can
#  be considered an unsupervised denoiser
from vq import VectorQuantizedVAE, VQVAEEmbeddings, VQVAEIntegerCodes  # noqa: F401
from vq.utils import VQEMAUpdater, VQLossHelper


def make_multimodal(
    n_samples: int = 512,
    n_components: int = 8,
    f_scale: float = 0.075,
) -> tuple[Tensor, Tensor]:
    """generate multimodal 2d data"""

    y = torch.arange(n_samples) % n_components

    # 2d rotation matrix
    ang = y * 2 * torch.pi / n_components
    c, s = torch.cos(ang), torch.sin(ang)
    R = torch.stack((c, s, -s, c), dim=-1).reshape(n_samples, 2, 2)

    eps = torch.randn(n_samples, 2) * f_scale

    # Are we sure we need 1 + 2*eps?
    X = torch.bmm(R, eps.add_(1 + eps).unsqueeze_(-1)).squeeze_(-1)
    return X, y


def split(
    *tensors: Tensor,
    fractions: tuple[float] = (0.5,),
) -> list[list[Tensor, ...], ...]:
    """Randomly split the provided tersors."""

    # make sure the tensors have identical first dim
    sizes = list(map(len, tensors))
    assert all(size == sizes[0] for size in sizes)

    # allow +ve fractions that sum to one at most
    assert all(s >= 0 for s in fractions)
    assert sum(fractions) <= 1

    # get the split slices
    total, j1, slices = sizes[0], 0, []
    for fraction in fractions:
        j0, j1 = j1, j1 + int(fraction * total)
        slices.append(slice(j0, j1))
    slices.append(slice(j1, None))

    permutation = torch.randperm(total)
    splits = [[t[permutation[sl]] for t in tensors] for sl in slices]
    return splits


# generate multimodal data
n_components = 8
X, y = make_multimodal(n_components=n_components)

# fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=120)
# ax.scatter(*X.T, c=y)
# ax.set_axis_off()
# fig.show()

# split and prep the datafeeds
feeds_specs = {
    "train": {"batch_size": 8, "shuffle": True},
    "test": {"batch_size": 128, "shuffle": False},  # UNUSED
}

splits, names = split(X, y, fractions=(0.75,)), ("train", "test")
datasets = {k: TensorDataset(*tt) for tt, k in zip(splits, names)}
feeds = {
    k: DataLoader(datasets[k], **spec) for k, spec in feeds_specs.items()
}

# training configuration and the model
n_epochs = 256
num_embeddings = 32  # number of clusters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mod = nn.Sequential(
    nn.Linear(2, 64, bias=False),
    nn.ReLU(),
    nn.Linear(64, 2, bias=False),

    # vq has special output format, so we use a handy wrapper
    VQVAEEmbeddings(VectorQuantizedVAE(num_embeddings, 2, -1)),
    nn.BatchNorm1d(2),
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, n_components),
).to(device)

# locate the VQ embeddings layer
index, vqref = next((
    (j, m.wrapped) for j, m in enumerate(mod) if isinstance(m, VQVAEEmbeddings)
), (1, None))

opt = torch.optim.Adam(mod.parameters(), lr=1e-3)

# this helps us extract and pull out the vq layer-specific losses
hlp = VQLossHelper(mod, reduction="sum")
ema = VQEMAUpdater(mod, alpha=0.25, update=True)

eval_cpu = tuple(datasets["test"].tensors)
(*eval_device,) = map(lambda x: x.to(device), eval_cpu)

hist = []
for ep in tqdm.tqdm(range(n_epochs), ncols=50):
    mod.train()
    for bx, by in iter(feeds["train"]):
        bx, by = bx.to(device), by.to(device)
        with hlp, ema:
            out = mod(bx)

        logits = out.log_softmax(dim=-1)

        clf_loss = F.nll_loss(logits, by)
        vq_ell = sum(hlp.finish().values())

        loss = clf_loss + vq_ell

        opt.zero_grad()
        # in this simple example ema updates render the vq_ell `term` non diffable
        if loss.grad_fn is not None:
            loss.backward()

        opt.step()

        ema.step()  # if ema were updating, then this would do the work!

    mod.eval()
    with torch.no_grad():
        out = mod(eval_device[0])
        y_pred = out.argmax(dim=-1).cpu()

        # intermediate representations
        rep = mod[:index](eval_device[0]).cpu()

    hist.append(
        (
            *ema.entropy.values(),
            float(vq_ell),
            float(clf_loss),
            float((eval_cpu[1] == y_pred).float().mean()),
        )
    )


## PLOTTING

fig, (ax, ax2) = plt.subplots(1, 2, dpi=120, figsize=(8, 2))
if vqref is not None:
    try:
        vor = Voronoi(vqref.weight.detach().cpu().numpy())
        voronoi_plot_2d(vor, ax=ax, show_vertices=False)
    except:
        pass
    ax.scatter(*vqref.weight.detach().cpu().numpy().T, s=5, color="C0")

ax.scatter(
    *rep.numpy().T,
    c=y_pred.numpy(),  # color='magenta',
    alpha=0.5,
    zorder=-10,
    s=5,
)
ax.set_aspect(1.0)
ax.set_axis_off()

*ents, ells, clfs, acc = map(np.array, zip(*hist))
if ents:
    l2, = ax2.plot(ents[0], label="entropy")

ax2_ = ax2.twinx()
l1, = ax2_.plot(acc, c="C2", label="accuracy")
if ents and False:
    l2, = ax2_.semilogy(ells, c="C1", label="vq-loss")

plt.show()
