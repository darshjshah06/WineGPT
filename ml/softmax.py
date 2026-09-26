import numpy as np
import pandas as pd

FEATURES = ["body_score", "acidity_score", "tannin", "sweetness", "abv"]


def load_wines(path="data/wines.csv", features=FEATURES):
    df = pd.read_csv(path)

    X = df[features].to_numpy(dtype=float)
    classes = sorted(df["type"].unique())
    y = np.array([classes.index(t) for t in df["type"]])

    return X, y, classes, df["name"].tolist()


class SoftmaxRegression:
    """Multinomial logistic regression trained with plain gradient descent."""

    def __init__(self, n_features, n_classes, l2=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (n_features, n_classes))
        self.b = np.zeros(n_classes)
        self.l2 = l2
        self.mean = None
        self.std = None

    def _standardize(self, X):
        return (X - self.mean) / self.std

    def probs(self, Xs):
        logits = Xs @ self.W + self.b
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def loss_and_grads(self, Xs, y):
        n = len(y)
        p = self.probs(Xs)

        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        loss += 0.5 * self.l2 * (self.W ** 2).sum()

        d = p.copy()
        d[np.arange(n), y] -= 1
        d /= n

        dW = Xs.T @ d + self.l2 * self.W
        db = d.sum(axis=0)

        return loss, [dW, db]

    def fit(self, X, y, optimizer, epochs=300, on_epoch=None):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)

        history = []

        for epoch in range(epochs):
            loss, grads = self.loss_and_grads(Xs, y)
            optimizer.step([self.W, self.b], grads)

            grad_norm = float(np.sqrt(sum((g ** 2).sum() for g in grads)))
            acc = float((self.probs(Xs).argmax(axis=1) == y).mean())
            history.append({
                "epoch": epoch,
                "loss": float(loss),
                "accuracy": acc,
                "grad_norm": grad_norm,
            })

            if on_epoch:
                on_epoch(history[-1])

        return history

    def predict_proba(self, X):
        return self.probs(self._standardize(X))
