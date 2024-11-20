import numpy as np
import cupy as cp

class Layer:
    def __init__(self, m, n_k_1, n_k, akt_func, xp=cp, dtype=cp.float32):
        self.xp = xp
        self.m = m
        self.w = xp.random.randn(n_k, n_k_1).astype(dtype) * 0.01
        self.b = xp.zeros((n_k, 1), dtype=dtype)
        self.A = xp.zeros((n_k_1, m), dtype=dtype)
        self.Z = xp.zeros((n_k, m), dtype=dtype)
        self.dZ = xp.zeros((n_k, m), dtype=dtype)
        self.dW = xp.zeros((n_k, n_k_1), dtype=dtype)
        self.db = xp.zeros((n_k, 1), dtype=dtype)
        self.akt_func = akt_func

    def forward_prog(self, A_k_1):
        self.Z = self.w @ A_k_1 + self.b
        self.A = self.akt_func(self.Z)

    def backword_prog_output(self, Y, A_k_1):
        self.dZ = self.A - Y
        self.dW = (1 / self.m) * self.dZ @ A_k_1.T
        self.db = (1 / self.m) * self.xp.sum(self.dZ, axis=1).reshape(-1, 1)

    def deriv_akt_func(self):
        return self.Z > 0

    def backward_prog(self, W_k_plus_1, dZ_k_plus_1, A_k_1):
        self.dZ = W_k_plus_1.T @ dZ_k_plus_1 * self.deriv_akt_func()
        self.dW = (1 / self.m) * self.dZ @ A_k_1.T
        self.db = (1 / self.m) * self.xp.sum(self.dZ, axis=1).reshape(-1, 1)

    def update(self, learning_rate):
        self.w -= learning_rate * self.dW
        self.b -= learning_rate * self.db