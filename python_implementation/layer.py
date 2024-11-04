import numpy as np
class Layer:
    
    def __init__(self, m, n_k_1, n_k, akt_func):
        self.m = m
        self.w = np.random.randn(n_k,n_k_1) *0.01
        self.b = np.zeros((n_k,1))
        self.A = np.zeros((n_k_1,m))
        self.Z = np.zeros((n_k,m))
        self.dZ = np.zeros((n_k,m))
        self.dW = np.zeros((n_k,n_k_1))
        self.db = np.zeros((n_k,1))
        self.akt_func = akt_func
        
    def forward_prog(self, A_k_1: np.array):
        self.Z = self.w @ A_k_1 + self.b
        self.A = self.akt_func(self.Z) 
    
    def backword_prog_output(self, Y:np.array, A_k_1:np.array):
        self.dZ = self.A - Y
        self.dW = (1/self.m) * self.dZ @ A_k_1.T
        self.db = (1/self.m) * np.sum(self.dZ, axis=1).reshape(-1, 1)
    
    def deriv_akt_func(self) -> np.array:
        return self.Z > 0
    
    def backward_prog(self, W_k_plus_1, dZ_k_plus_1, A_k_1):
        self.dZ = W_k_plus_1.T @ dZ_k_plus_1 * self.deriv_akt_func()
        self.dW = (1/self.m) * self.dZ @ A_k_1.T
        self.db = (1/self.m) * np.sum(self.dZ, axis=1).reshape(-1, 1)
        
    def update(self, learning_rate):
        self.w = self.w - learning_rate*self.dW
        self.b = self.b - learning_rate*self.db
        