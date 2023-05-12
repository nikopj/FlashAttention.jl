using NNlib
using Test
using FlashAttention

@testset "naive vs flash vs NNlib" begin
Nkv = 30
Nq  = 30
dqk = 12
dv  = 6
bs  = 2

q, k, v = rand(Nq, dqk, bs), rand(Nkv, dqk, bs), rand(Nkv, dv, bs)

y0, P0 = dot_product_attention( permutedims(q, (2, 1, 3)), permutedims(k, (2, 1, 3)), permutedims(v, (2, 1, 3)))

y1, P1 = dense_dpa(q, k, v)
y2 = first(dense_fa(q, k, v))

@test y1 ≈ permutedims(y0, (2, 1, 3))
@test y2 ≈ y1
end

@btime dot_product_attention( $(permutedims(q, (2, 1, 3))), $(permutedims(k, (2, 1, 3))), $(permutedims(v, (2, 1, 3))))
@btime dense_dpa(q, k, v)
@btime dense_fa(q, k, v)

1+1

