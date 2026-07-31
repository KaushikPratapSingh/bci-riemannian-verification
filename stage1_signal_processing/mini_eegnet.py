"""
A minimal, from-scratch 1D convolutional network -- because this sandbox has
no PyTorch or TensorFlow available (no network access to install them).

This is implemented the same way SOBI was implemented earlier in this
project: by hand, then verified with a rigorous check BEFORE being trusted
on real data. For SOBI the check was a known-answer separation test. For a
from-scratch backprop implementation, the standard, gold-standard check is
a numerical gradient check (finite differences vs. the analytical gradient).
If that check fails, nothing downstream can be trusted, no matter how
plausible the training curve looks.

Architecture (deliberately small, matching an EEGNet-style design pattern
at a scale appropriate to a 4-channel, small-sample problem):
  Conv1D(4 -> 8 filters, kernel=25)  ->  ReLU  ->  GlobalAveragePool  ->  Dense(8 -> 1)
"""

import numpy as np


class Conv1D:
    def __init__(self, c_in, c_out, kernel_size, seed=0):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (c_in * kernel_size))
        self.W = rng.standard_normal((kernel_size, c_in, c_out)) * scale
        self.b = np.zeros(c_out)
        self.K, self.c_in, self.c_out = kernel_size, c_in, c_out
        self._cache = None

    def forward(self, X):
        """X: (T, c_in) -> Y: (T-K+1, c_out)"""
        T = X.shape[0]
        T_out = T - self.K + 1
        # Build sliding-window view (T_out, K, c_in) without a manual loop
        windows = np.stack([X[i:i + self.K] for i in range(T_out)], axis=0)
        Y = np.einsum('tkc,kco->to', windows, self.W) + self.b
        self._cache = (X, windows)
        return Y

    def backward(self, dY, lr):
        X, windows = self._cache
        T_out = dY.shape[0]
        dW = np.einsum('tkc,to->kco', windows, dY)
        db = np.sum(dY, axis=0)
        dX = np.zeros_like(X)
        for i in range(T_out):
            dX[i:i + self.K] += np.einsum('co,kco->kc', dY[i], self.W)
        self.W -= lr * dW
        self.b -= lr * db
        return dX

    def get_params_grads(self, dY):
        X, windows = self._cache
        dW = np.einsum('tkc,to->kco', windows, dY)
        db = np.sum(dY, axis=0)
        return [self.W, self.b], [dW, db]


class ReLU:
    def __init__(self):
        self._cache = None

    def forward(self, X):
        self._cache = X
        return np.maximum(0, X)

    def backward(self, dY):
        X = self._cache
        return dY * (X > 0)


class GlobalAvgPool:
    def __init__(self):
        self._t = None

    def forward(self, X):
        self._t = X.shape[0]
        return X.mean(axis=0)

    def backward(self, dY):
        return np.tile(dY / self._t, (self._t, 1))


class Dense:
    def __init__(self, n_in, n_out, seed=1):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(n_out)
        self._cache = None

    def forward(self, x):
        self._cache = x
        return self.W @ x + self.b

    def backward(self, dy, lr):
        x = self._cache
        dW = np.outer(dy, x)
        db = dy
        dx = self.W.T @ dy
        self.W -= lr * dW
        self.b -= lr * db
        return dx

    def get_params_grads(self, dy):
        x = self._cache
        dW = np.outer(dy, x)
        db = dy
        return [self.W, self.b], [dW, db]


class MiniEEGNet:
    """Conv1D -> ReLU -> GlobalAvgPool -> Dense(->1), trained with plain SGD."""

    def __init__(self, c_in=4, n_filters=8, kernel_size=25, seed=0):
        self.conv = Conv1D(c_in, n_filters, kernel_size, seed=seed)
        self.relu = ReLU()
        self.pool = GlobalAvgPool()
        self.dense = Dense(n_filters, 1, seed=seed + 1)

    def forward(self, X):
        """X: (T, c_in) raw window -> scalar prediction."""
        h = self.conv.forward(X)
        h = self.relu.forward(h)
        h = self.pool.forward(h)
        y = self.dense.forward(h)
        return y[0]

    def backward_and_step(self, dy, lr):
        _, (dW_dense, db_dense) = self.dense.get_params_grads(np.array([dy]))
        dh = self.dense.W.T @ np.array([dy])
        self.dense.W -= lr * dW_dense
        self.dense.b -= lr * db_dense

        dh_pool = self.pool.backward(dh)
        dh_relu = self.relu.backward(dh_pool)
        _, (dW_conv, db_conv) = self.conv.get_params_grads(dh_relu)
        self.conv.W -= lr * dW_conv
        self.conv.b -= lr * db_conv

    def get_flat_params(self):
        return [self.conv.W, self.conv.b, self.dense.W, self.dense.b]

    def set_flat_params(self, params):
        self.conv.W, self.conv.b, self.dense.W, self.dense.b = params

    def train_step(self, X, y_true, lr):
        y_pred = self.forward(X)
        loss = (y_pred - y_true) ** 2
        dy = 2 * (y_pred - y_true)
        self.backward_and_step(dy, lr)
        return loss, y_pred


def numerical_gradient_check():
    """The mandatory sanity check before this implementation is trusted on
    any real task. Compares the analytical backprop gradient against a
    finite-difference numerical gradient for every parameter tensor."""
    rng = np.random.default_rng(42)
    net = MiniEEGNet(c_in=4, n_filters=3, kernel_size=5, seed=0)
    X = rng.standard_normal((40, 4))
    y_true = 0.7

    y_pred = net.forward(X)
    loss = (y_pred - y_true) ** 2
    dy = 2 * (y_pred - y_true)

    # Analytical gradients
    dh = net.dense.W.T @ np.array([dy])  # before zeroing dense, grab dense grads first
    _, (dW_dense, db_dense) = net.dense.get_params_grads(np.array([dy]))
    dh_pool = net.pool.backward(dh)
    dh_relu = net.relu.backward(dh_pool)
    _, (dW_conv, db_conv) = net.conv.get_params_grads(dh_relu)

    eps = 1e-5
    max_rel_err = 0.0

    def loss_with_current_params():
        yp = net.forward(X)
        return (yp - y_true) ** 2

    checks = [
        ("conv.W", net.conv.W, dW_conv),
        ("conv.b", net.conv.b, db_conv),
        ("dense.W", net.dense.W, dW_dense),
        ("dense.b", net.dense.b, db_dense),
    ]

    print("Numerical gradient check (analytical vs. finite-difference):")
    for name, param, analytical_grad in checks:
        flat_param = param.reshape(-1)
        flat_grad = analytical_grad.reshape(-1)
        n_check = min(5, len(flat_param))  # spot-check a handful of entries per tensor
        idxs = np.linspace(0, len(flat_param) - 1, n_check).astype(int)
        for idx in idxs:
            orig = flat_param[idx]
            flat_param[idx] = orig + eps
            loss_plus = loss_with_current_params()
            flat_param[idx] = orig - eps
            loss_minus = loss_with_current_params()
            flat_param[idx] = orig  # restore exactly
            numerical_grad = (loss_plus - loss_minus) / (2 * eps)
            rel_err = abs(numerical_grad - flat_grad[idx]) / (abs(numerical_grad) + abs(flat_grad[idx]) + 1e-12)
            max_rel_err = max(max_rel_err, rel_err)
        print(f"  {name:<10} spot-checked {n_check} of {len(flat_param)} parameters")
    print(f"  OVERALL max relative error: {max_rel_err:.2e}  "
          f"({'PASS' if max_rel_err < 1e-3 else 'FAIL'}, threshold 1e-3)")
    return max_rel_err < 1e-3


if __name__ == "__main__":
    ok = numerical_gradient_check()
    if not ok:
        raise SystemExit("Gradient check failed -- do not trust this implementation on real data.")
