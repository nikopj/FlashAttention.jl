module FlashAttention

using Base.Threads
using LoopVectorization
using LinearAlgebra, SparseArrays
using NNlib, MLUtils
# using CUDA, CUDA.CUSPARSE

include("softmax.jl")
include("circulant.jl")
include("naive.jl")
export dense_dpa, windowed_dpa, circulant_dpa

include("flash.jl")
export dense_fa, circulant_fa

end
