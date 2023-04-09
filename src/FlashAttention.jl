module FlashAttention

using Base.Threads
using LinearAlgebra, SparseArrays
using NNlib, MLUtils
# using CUDA, CUDA.CUSPARSE

# include("circulant.jl")
include("naive.jl")
export dense_dpa, windowed_dpa

include("flash.jl")
export dense_fa

end
