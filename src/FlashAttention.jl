module FlashAttention

using Base.Threads
using LoopVectorization
using LinearAlgebra, SparseArrays
using NNlib, NNlibCUDA
using MLUtils
using CUDA

include("softmax.jl")
include("circulant.jl")
include("naive.jl")
export dense_dpa, windowed_dpa, circulant_dpa

include("flash.jl")
include("cuda/flash.jl")
export dense_fa, circulant_fa

end
