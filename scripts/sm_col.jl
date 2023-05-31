using FlashAttention: sm_naive!, fused_softmax!
using NNlib, NNlibCUDA
using CUDA
using BenchmarkTools

V = CUDA.randn(2^10, 2^11)

U = sm_naive!(similar(V), V)
Q = fused_softmax!(CUDA.randn(size(V)...), V)
W = NNlib.softmax!(similar(V), V)
@show U ≈ W
@show Q ≈ W

# kb1 = @benchmark CUDA.@sync sm_naive!($U, $V)
# kb2 = @benchmark CUDA.@sync fused_softmax!($Q, $V)
# kb3 = @benchmark CUDA.@sync NNlib.softmax!($W, $V);
# k
# k@show bw1 = 4*5*(prod∘size)(V) / median(b1).time  # GB/s

