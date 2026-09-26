"""Animated gradient descent dashboard.

    python -m ml.visualize                    # opens an animated window
    python -m ml.visualize --surface bumpy    # rosenbrock | himmelblau | saddle | bumpy
    python -m ml.visualize --save lab.gif     # export a GIF instead
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from ml.optimizers import OPTIMIZERS
from ml.softmax import SoftmaxRegression, load_wines

BG = "#0c080a"
PANEL = "#150e12"
TEXT = "#f3e9ec"
MUTED = "#a08c94"
COLORS = {"sgd": "#38bdf8", "momentum": "#4ade80", "rmsprop": "#e879f9", "adam": "#fde68a"}
CLASS_COLORS = {"red": "#d7264f", "rose": "#f59bb5", "sparkling": "#7cc9ee", "white": "#eed98a"}
WINE_CMAP = LinearSegmentedColormap.from_list("wine", [
    "#06030a", "#1a0a2c", "#3a0e4d", "#651456", "#932052",
    "#c03c44", "#e06a39", "#f39f55", "#fbd394", "#fff3dc",
])

SURFACES = {
    "rosenbrock": dict(
        title="Rosenbrock (banana valley)",
        f=lambda x, y: (1 - x) ** 2 + 100 * (y - x ** 2) ** 2,
        g=lambda x, y: np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)]),
        xr=(-2, 2), yr=(-1, 3), start=(-1.6, 2.6), minima=[(1, 1)],
        lr=dict(sgd=0.0008, momentum=0.00025, rmsprop=0.012, adam=0.04),
    ),
    "himmelblau": dict(
        title="Himmelblau (four valleys)",
        f=lambda x, y: (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2,
        g=lambda x, y: np.array([
            4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7),
            2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7),
        ]),
        xr=(-5, 5), yr=(-5, 5), start=(-4.5, 0.2),
        minima=[(3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)],
        lr=dict(sgd=0.008, momentum=0.0015, rmsprop=0.04, adam=0.12),
    ),
    "saddle": dict(
        title="Saddle point",
        f=lambda x, y: 0.3 * x ** 2 + 0.25 * y ** 4 - y ** 2 + 2,
        g=lambda x, y: np.array([0.6 * x, y ** 3 - 2 * y]),
        xr=(-4, 4), yr=(-2.6, 2.6), start=(-3.6, 0.004), minima=[(0, 1.414), (0, -1.414)],
        lr=dict(sgd=0.04, momentum=0.008, rmsprop=0.02, adam=0.05),
    ),
    "bumpy": dict(
        title="Bumpy bowl (local minima)",
        f=lambda x, y: 0.1 * (x ** 2 + y ** 2) + 0.25 * (2 - np.cos(2.5 * x) - np.cos(2.5 * y)),
        g=lambda x, y: np.array([0.2 * x + 0.625 * np.sin(2.5 * x), 0.2 * y + 0.625 * np.sin(2.5 * y)]),
        xr=(-4.5, 4.5), yr=(-4.5, 4.5), start=(3.9, 3.1), minima=[(0, 0)],
        lr=dict(sgd=0.05, momentum=0.04, rmsprop=0.05, adam=0.3),
    ),
}


def race(surface, steps):
    S = SURFACES[surface]
    paths, losses = {}, {}

    for name, cls in OPTIMIZERS.items():
        p = np.array(S["start"], dtype=float)
        opt = cls(lr=S["lr"][name])
        path, loss = [p.copy()], [S["f"](*p)]

        for _ in range(steps):
            opt.step([p], [S["g"](*p)])
            if not np.all(np.isfinite(p)) or np.abs(p).max() > 1e4:
                break
            path.append(p.copy())
            loss.append(S["f"](*p))

        paths[name] = np.array(path)
        losses[name] = np.array(loss)

    return paths, losses


def poly(X):
    a, b = X[:, 0], X[:, 1]
    return np.column_stack([a, b, a * a, b * b, a * b])


def train_cellar(epochs):
    X, y, classes, _ = load_wines(features=["body_score", "acidity_score"])
    Xp = poly(X)
    model = SoftmaxRegression(Xp.shape[1], len(classes))
    snapshots = []

    def snap(h):
        snapshots.append((model.W.copy(), model.b.copy(), h))

    model.fit(Xp, y, OPTIMIZERS["adam"](lr=0.05), epochs=epochs, on_epoch=snap)
    return X, y, classes, model, snapshots


def style(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=12, loc="left", pad=10, fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2d1f26")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--surface", choices=SURFACES, default="rosenbrock")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--save", metavar="FILE", help="write the animation to a .gif instead of opening a window")
    args = parser.parse_args()

    S = SURFACES[args.surface]
    paths, losses = race(args.surface, args.steps)
    X, y, classes, model, snapshots = train_cellar(args.epochs)

    plt.rcParams.update({"font.family": "sans-serif", "text.color": TEXT})
    fig = plt.figure(figsize=(14, 8.5), facecolor=BG)
    fig.suptitle("WineGPT · Gradient Lab", color=TEXT, fontsize=18, fontweight="bold", x=0.03, ha="left", y=0.975)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.32, wspace=0.18,
                          left=0.04, right=0.98, top=0.9, bottom=0.07)
    ax_surf = fig.add_subplot(gs[:, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_cell = fig.add_subplot(gs[1, 1])

    # --- loss surface
    style(ax_surf, S["title"])
    gx = np.linspace(*S["xr"], 400)
    gy = np.linspace(*S["yr"], 400)
    XX, YY = np.meshgrid(gx, gy)
    Z = S["f"](XX, YY)
    Zn = np.log1p(Z - Z.min())
    ax_surf.contourf(XX, YY, Zn, levels=60, cmap=WINE_CMAP)
    ax_surf.contour(XX, YY, Zn, levels=24, colors="white", linewidths=0.4, alpha=0.35)
    for mx, my in S["minima"]:
        ax_surf.plot(mx, my, "w+", ms=12, mew=1.5)
    ax_surf.plot(*S["start"], "o", color="white", ms=5)
    ax_surf.set_xlim(S["xr"])
    ax_surf.set_ylim(S["yr"])

    lines, heads = {}, {}
    for name, c in COLORS.items():
        (lines[name],) = ax_surf.plot([], [], color=c, lw=2, alpha=0.9, label=name)
        (heads[name],) = ax_surf.plot([], [], "o", color=c, ms=9, mec=BG, mew=1.5)
    leg = ax_surf.legend(loc="lower right", facecolor=BG, edgecolor="#2d1f26", labelcolor=TEXT, fontsize=10)
    step_text = ax_surf.text(0.02, 0.97, "", transform=ax_surf.transAxes, color=TEXT, va="top",
                             family="monospace", fontsize=10,
                             bbox=dict(facecolor=BG, edgecolor="#2d1f26", alpha=0.85, boxstyle="round"))
    leg.set_zorder(5)

    # --- race loss curves
    style(ax_loss, "Loss per optimizer (log scale)")
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(0, args.steps)
    all_losses = np.concatenate([l[np.isfinite(l)] for l in losses.values()])
    ax_loss.set_ylim(max(all_losses.min(), 1e-8), all_losses.max() * 2)
    ax_loss.grid(color="#ffffff", alpha=0.06)
    loss_lines = {n: ax_loss.plot([], [], color=c, lw=1.8)[0] for n, c in COLORS.items()}

    # --- cellar decision boundaries
    style(ax_cell, "")
    pad = 0.3
    cx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 160)
    cy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 120)
    CX, CY = np.meshgrid(cx, cy)
    grid = model._standardize(poly(np.column_stack([CX.ravel(), CY.ravel()])))
    cmap = ListedColormap([CLASS_COLORS[c] for c in classes])
    region = ax_cell.imshow(np.zeros_like(CX), extent=(cx[0], cx[-1], cy[0], cy[-1]), origin="lower",
                            cmap=cmap, vmin=0, vmax=len(classes) - 1, alpha=0.35, aspect="auto",
                            interpolation="bilinear")
    for k, c in enumerate(classes):
        m = y == k
        ax_cell.scatter(X[m, 0], X[m, 1], s=46, color=CLASS_COLORS[c], edgecolor=BG, lw=1.2, label=c, zorder=3)
    ax_cell.set_xlabel("body", color=MUTED)
    ax_cell.set_ylabel("acidity", color=MUTED)
    ax_cell.legend(loc="upper right", facecolor=BG, edgecolor="#2d1f26", labelcolor=TEXT, fontsize=8, ncol=2)

    frames = 180
    race_len = max(len(p) for p in paths.values())

    def update(i):
        t = (i + 1) / frames
        k = max(1, int(t * race_len))
        for name in COLORS:
            p = paths[name][:k]
            lines[name].set_data(p[:, 0], p[:, 1])
            heads[name].set_data([p[-1, 0]], [p[-1, 1]])
            loss_lines[name].set_data(np.arange(min(k, len(losses[name]))), losses[name][:k])
        step_text.set_text(f"step {k - 1}   θ ← θ − η∇L")

        W, b, h = snapshots[max(0, int(t * len(snapshots)) - 1)]
        probs = grid @ W + b
        region.set_data(probs.argmax(axis=1).reshape(CX.shape))
        ax_cell.set_title(
            f"Wine classifier · epoch {h['epoch']}  loss {h['loss']:.3f}  acc {h['accuracy']:.0%}",
            color=TEXT, fontsize=12, loc="left", pad=10, fontweight="bold",
        )
        return [*lines.values(), *heads.values(), *loss_lines.values(), region, step_text]

    anim = FuncAnimation(fig, update, frames=frames, interval=33, repeat_delay=2500)

    if args.save:
        anim.save(args.save, writer="pillow", fps=30, savefig_kwargs={"facecolor": BG})
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
