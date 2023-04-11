using BenchmarkTools
using FlashAttention

N = 2048
d = 64
batchsize = 2
q, k, v = rand(N, d, batchsize), rand(N, d, batchsize), rand(N, d, batchsize)

@show (N, d, batchsize)

M = 32000

Bc = clamp(ceil(Int, M/d), 1, N)
Br = clamp(min(d, ceil(Int, M/2d)), 1, N)
Bb = 1
Tb = cld(batchsize, Bb)
Tr = cld(N, Br)
Tc = cld(N, Bc)

@show (Bb, Br, Bc) (Tb, Tr, Tc)
mem(r,c,b,d) = 6*r + 3*r*d + 2*c*d + r*c
@show mem(Br, Bc, Bb, d)

y = first(dense_dpa(q,k,v))
@show y ≈ first(dense_fa(q,k,v))

@btime dense_dpa(q, k, v)
@btime dense_fa(q, k, v)

1+1

