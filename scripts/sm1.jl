using FlashAttention: sm_naive!, fused_softmax!
using NNlib, NNlibCUDA
using CUDA
using BenchmarkTools

v = CUDA.randn(2^24)

u = sm_naive!(similar(v), v)
q = fused_softmax!(similar(v))
w = NNlib.softmax!(similar(v), v)
@show u ≈ w
@show q ≈ w

b1 = @benchmark CUDA.@sync sm_naive!($u, $v)
b2 = @benchmark CUDA.@sync fused_softmax!($q)
b3 = @benchmark CUDA.@sync NNlib.softmax!($w, $v)

