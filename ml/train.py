"""Train a wine-type classifier with gradient descent and watch the loss fall.

    python -m ml.train --optimizer adam --lr 0.05 --epochs 300
    python -m ml.train --race
"""

import argparse

import numpy as np

from ml.optimizers import OPTIMIZERS
from ml.softmax import SoftmaxRegression, load_wines

DEFAULT_LR = {"sgd": 0.5, "momentum": 0.1, "rmsprop": 0.02, "adam": 0.05}
BARS = "▁▂▃▄▅▆▇█"
WINE = "\033[38;5;161m"
DIM = "\033[2m"
RESET = "\033[0m"


def sparkline(values, width=60):
    idx = np.linspace(0, len(values) - 1, min(width, len(values))).astype(int)
    v = np.array(values)[idx]
    lo, hi = v.min(), v.max()
    scaled = (v - lo) / (hi - lo + 1e-12)
    return "".join(BARS[int(s * (len(BARS) - 1))] for s in scaled)


def train(name, lr, epochs, verbose=True):
    X, y, classes, _ = load_wines()
    model = SoftmaxRegression(X.shape[1], len(classes))
    optimizer = OPTIMIZERS[name](lr=lr)

    def log(h):
        if verbose and (h["epoch"] % max(1, epochs // 10) == 0 or h["epoch"] == epochs - 1):
            bar = "█" * int(h["accuracy"] * 30)
            print(
                f"  epoch {h['epoch']:>4}  loss {h['loss']:.4f}  "
                f"|∇| {h['grad_norm']:.4f}  acc {WINE}{bar:<30}{RESET} {h['accuracy']:.0%}"
            )

    history = model.fit(X, y, optimizer, epochs=epochs, on_epoch=log)
    return model, history, classes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adam")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--race", action="store_true", help="train with every optimizer and compare")
    args = parser.parse_args()

    if args.race:
        print(f"\n{WINE}🍷 optimizer race{RESET} {DIM}({args.epochs} epochs, softmax regression on {len(load_wines()[1])} wines){RESET}\n")
        for name in OPTIMIZERS:
            _, history, _ = train(name, DEFAULT_LR[name], args.epochs, verbose=False)
            losses = [h["loss"] for h in history]
            print(f"  {name:<9} {WINE}{sparkline(losses)}{RESET}  final loss {losses[-1]:.4f}  acc {history[-1]['accuracy']:.0%}")
        print()
        return

    lr = args.lr or DEFAULT_LR[args.optimizer]
    print(f"\n{WINE}🍷 training wine-type classifier{RESET} {DIM}optimizer={args.optimizer} lr={lr}{RESET}\n")
    model, history, classes = train(args.optimizer, lr, args.epochs)

    losses = [h["loss"] for h in history]
    print(f"\n  loss  {WINE}{sparkline(losses)}{RESET}")
    print(f"  {DIM}{losses[0]:.3f} → {losses[-1]:.3f}{RESET}\n")


if __name__ == "__main__":
    main()
