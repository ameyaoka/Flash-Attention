from tinygrad.tensor import Tensor

def flash_attention(Q, K, V, M=128):
    N, d = Q.shape
    O = Tensor.zeros(Q.shape).contiguous()  # Output tensor
    L = Tensor.full((N, 1), -1e9).contiguous()  # Track max values per query
    S = Tensor.zeros((N, 1)).contiguous()       # Track sum of exponents per query

    for i in range(0, N, M):
        Q_tile = Q[i:i+M].contiguous()
        mi = Tensor.full((Q_tile.shape[0], 1), -1e9).contiguous()  # Local max
        li = Tensor.zeros((Q_tile.shape[0], 1)).contiguous()        # Local sum

        for j in range(0, N, M):
            K_tile = K[j:j+M].contiguous()
            V_tile = V[j:j+M].contiguous()

            # Compute attention scores for current tile
            S_tile = (Q_tile @ K_tile.transpose()) / (d ** 0.5)

            # Update local max and sum
            mi_new = Tensor.maximum(mi, S_tile.max(axis=1, keepdim=True))
            exp_S_tile = (S_tile - mi_new).exp()
            li_new = (li * (mi - mi_new).exp()) + exp_S_tile.sum(axis=1, keepdim=True)

            # Update output with scaled values (critical fix)
            scale = (mi - mi_new).exp()
            O[i:i+M] = O[i:i+M] * scale + (exp_S_tile @ V_tile)
            
            mi = mi_new
            li = li_new

        # Save final max and sum for normalization
        L[i:i+M] = mi
        S[i:i+M] = li

    # Final normalization (remove extra term)
    O = O / S
    return O
