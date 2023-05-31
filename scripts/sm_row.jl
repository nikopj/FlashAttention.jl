using FlashAttention: sm_naive!, fused_softmax!
using NNlib, NNlibCUDA
using CUDA
using BenchmarkTools

V = CUDA.randn(2^16, 2^12)

U = sm_naive!(similar(V), V; dims=2)
Q = fused_softmax!(similar(V), V; dims=2)
W = NNlib.softmax!(similar(V), V; dims=2)
@show U ≈ W
@show Q ≈ W

b1 = @benchmark CUDA.@sync sm_naive!($U, $V; dims=2)
b2 = @benchmark CUDA.@sync fused_softmax!($Q, $V; dims=2)
b3 = @benchmark CUDA.@sync NNlib.softmax!($W, $V; dims=2);
