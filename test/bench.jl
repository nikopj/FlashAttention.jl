
N = 128
d = 64
bs = 2
q, k, v = rand(N, d, bs), rand(N, d, bs), rand(N, d, bs)

@btime y1 = first(dense_dpa(q, k, v));
@btime y2 = first(dense_fa(q, k, v));


