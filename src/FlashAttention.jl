module FlashAttention

using LinearAlgebra, NNlib
using MLUtils

include("naive.jl")
export dense_dpa, windowed_dpa

end
