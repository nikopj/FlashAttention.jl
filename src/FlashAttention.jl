module FlashAttention

using Base.Threads
using LoopVectorization
using LinearAlgebra, SparseArrays
using SparseMatricesCSR
using NNlib, NNlibCUDA
using MLUtils
using CUDA
using CUDA: i32

include("fused_softmax.jl")
export fused_softmax

include("utils.jl")

include("naive/dense.jl")
include("naive/windowed.jl")
include("naive/circulant.jl")
export dense_dpa!, circulant_dpa!
export dense_dpa, windowed_dpa, circulant_dpa, block_dpa

include("dense.jl")
include("windowed.jl")
include("circulant.jl")
export dense_fa!, circulant_fa!
export dense_fa, windowed_fa, circulant_fa, block_fa

#include("cuda/flash.jl")

end
