using FlashAttention: sm_naive!, fused_softmax!
using NNlib, NNlibCUDA
using CUDA
using BenchmarkTools

v = CUDA.randn(2^24)

u = sm_naive!(similar(v), v)
q = fused_softmax!(similar(v), v)
w = NNlib.softmax!(similar(reshape(v, :, 1)), reshape(v, :, 1))
@show u ≈ w
@show q ≈ w

b1 = @benchmark CUDA.@sync sm_naive!($u, $v)
b2 = @benchmark CUDA.@sync fused_softmax!($q, $v)
b3 = @benchmark CUDA.@sync NNlib.softmax!($w, $(reshape(v, :, 1)));

@show bw1 = 4*5*length(v) / median(b1).time  # GB/s

