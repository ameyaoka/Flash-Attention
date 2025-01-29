from tinygrad.tensor import Tensor

class FlashAttention:
    def __init__(self, head_dim, tile_size=128):
        self.head_dim = head_dim
        self.tile_size = tile_size  # Optional: Auto-select tile size if needed

    def __call__(self, Q, K, V):
        """FlashAttention forward pass."""
        N, d = Q.shape
        O = Tensor.zeros(Q.shape)
        L = Tensor.full((N, 1), -1e9)
        S = Tensor.zeros((N, 1))

        # Split into tiles
        for i in range(0, N, self.tile_size):
            Q_tile = Q[i:i+self.tile_size]
            mi = Tensor.full((Q_tile.shape[0], 1), -1e9)
            li = Tensor.zeros((Q_tile.shape[0], 1))

            for j in range(0, N, self.tile_size):
                K_tile = K[j:j+self.tile_size]
                V_tile = V[j:j+self.tile_size]

                # Scaled dot-product attention
                S_tile = (Q_tile @ K_tile.transpose()) / (self.head_dim ** 0.5)
                mi_new = Tensor.maximum(mi, S_tile.max(axis=1, keepdim=True))
                exp_S_tile = (S_tile - mi_new).exp()
                li_new = (li * (mi - mi_new).exp()) + exp_S_tile.sum(axis=1, keepdim=True)

                # Update output
                scale = (mi - mi_new).exp()
                O[i:i+self.tile_size] = O[i:i+self.tile_size] * scale + (exp_S_tile @ V_tile)
                mi, li = mi_new, li_new

            L[i:i+self.tile_size] = mi
            S[i:i+self.tile_size] = li

        O = O / S
        return O



