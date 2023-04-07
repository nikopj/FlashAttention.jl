# FlashAttention.jl
Julia implementation of flash-attention operation for neural networks.

### Recommended Julia Install Method
Install Julia via the [juliaup](https://github.com/JuliaLang/juliaup) manager:
```bash
curl -fsSL https://install.julialang.org | sh
```

## Todo
### Julia implementation

**Naive algorithms**:
The following are naive in the sense that they all form the normalized adjacency matrix `P`.

- [x] naive dot-product attention (`dense_dpa`) (1D/2D/3D)
- [x] naive block-sparse attention (`windowed_dpa` w/ `stride=windowsize`) (1D/2D/3D)
- [x] naive windowed sparse attention (`windowed_dpa`) (1D/2D/3D)
- [ ] naive circulant sparse attention (`circulant_dpa`) (1D/2D)

**Flash-Attention algorithms**:
- [ ] dense flash-attention (`dense_fa`) (1D/2D/3D)
- [ ] block-sparse flash-attention (`block_fa`) (1D/2D)
- [ ] circulant flash-attention (`circulant_fa`) (1D/2D)

### C implementation

