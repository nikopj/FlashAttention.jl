# FlashAttention.jl
Julia implementation of [Flash-Attention](https://github.com/HazyResearch/flash-attention),
a fused kernel for dot-product attention operation in transformer neural networks.

### Recommended Julia Install Method
Install Julia via the [juliaup](https://github.com/JuliaLang/juliaup) manager:
```bash
curl -fsSL https://install.julialang.org | sh
```

## Todo
### Julia implementation

**Naive algorithms**:
The following are naive in the sense that they all form the normalized adjacency matrix `P`.
Note that GPU version come for free-ish from all of these naive algorithms by making use of NNlib.jl.

- [x] naive dot-product attention (`dense_dpa`) 
    * [x] 1D/2D/3D
    * [x] backward (needs testing for correctness)
- [x] naive windowed sparse attention (`windowed_dpa`) 
    * [x] 1D/2D/3D
    * [x] backward
- [ ] naive circulant sparse attention (`circulant_dpa`) 
    * [x] 1D
    * [ ] 2D
    * [ ] backward

**Flash-Attention algorithms**:

*CPU*:
- [x] dense flash-attention (`dense_fa`) 
    * [x] 1D/2D/3D
    * [x] backward (needs testing for correctness)
- [x] windowed-sparse flash-attention (`windowed_fa`) 
    * [x] 1D/2D/3D
    * [x] backward
- [ ] circulant flash-attention (`circulant_fa`) (1D/2D)
    * [x] 1D
    * [ ] 2D
    * [ ] backward
- [ ] benchmark against naive versions

*GPU*:
- [ ] dense flash-attention (`dense_fa`) 
    * [x] 1D/2D/3D
    * [ ] backward (needs testing)
- [ ] block-sparse flash-attention (`block_fa`) 
    * [ ] 1D/2D
    * [ ] backward
- [ ] circulant flash-attention (`circulant_fa`) (1D/2D)
    * [ ] 1D
    * [ ] 2D
    * [ ] backward
- [ ] benchmark against naive versions

