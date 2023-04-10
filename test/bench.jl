using BenchmarkTools

N = 25^2
d = 64
bs = 16
q, k, v = rand(N, d, bs), rand(N, d, bs), rand(N, d, bs)

M = 1024000
# row/column block-length
Br = min(d, cld(M, 4*d*bs))
Bc = cld(M, 4*d*bs)
# num row/column blocks
Tr = cld(N, Br)
Tc = cld(N, Bc)

@show (Br, Bc) (Tr, Tc)
@show first(dense_dpa(q,k,v)) ≈ first(dense_fa(q,k,v))

@btime dense_dpa(q, k, v);
@btime dense_fa(q, k, v);

