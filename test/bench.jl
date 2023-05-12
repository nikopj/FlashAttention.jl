using BenchmarkTools
using FlashAttention

W = 128
N = 1024
d = 64
batchsize = 32
q = rand(N, d, batchsize)
k = rand(N, d, batchsize)
v = rand(N, d, batchsize)

@show (N, d, batchsize)

M = 32000

Bc = clamp(ceil(Int, M/d), 1, N)
Br = clamp(min(d, ceil(Int, M/2d)), 1, N)
Bb = 1
Tb = cld(batchsize, Bb)
Tr = cld(N, Br)
Tc = cld(N, Bc)
@show (Bb, Br, Bc) (Tb, Tr, Tc)

# Bw = clamp(ceil(Int, M/d), 1, W)
# #Br = clamp(min(d, ceil(Int, M/d)), 1, N)
# Br = clamp(ceil(Int, M/8d), 1, N)
# Bb = 1
# Tb = cld(batchsize, Bb)
# Tr = cld(N, Br)
# Tw = cld(W, Bw)
# @show (Bb, Br, Bw) (Tb, Tr, Tw)

# mem(r,c,b,d) = 6*r + 3*r*d + 2*c*d + r*c
# @show mem(Br, Bc, Bb, d)

y = first(dense_dpa(q,k,v))
@show y ≈ first(dense_fa(q,k,v))

@btime dense_dpa(q, k, v)
@btime dense_fa(q, k, v)

# y = first(circulant_dpa(q, k, v, W))
# @show y ≈ first(circulant_fa(q, k, v, W))

#@btime circulant_dpa(q, k, v, W)
# @btime circulant_fa(q, k, v, W)

1+1
