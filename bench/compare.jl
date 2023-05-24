using FlashAttention
using Printf
using Test

function time_dense(N, d, bs; reps=100, T=Float64)
    shape = (N, d, bs)
    # init variables
    Q = randn(T, shape)
    K = randn(T, shape)
    V = randn(T, shape)
    O1 = similar(Q, shape)
    O2 = similar(Q, shape)
    P = similar(Q, N, N, bs)
    l = similar(Q, N, 1, bs)
    m = similar(Q, N, 1, bs)

    # warmup
    dense_dpa!(O1, P, Q, K, V)
    dense_fa!(O2, l, m, Q, K, V)
    @test O1 ≈ O2

    time_dpa = 0.0
    time_fa  = 0.0
    for i=1:reps
        time_dpa += @elapsed dense_dpa!(O1, P, Q, K, V)
        time_fa  += @elapsed  dense_fa!(O2, l, m, Q, K, V)
    end

    return time_dpa/reps, time_fa/reps
end

function time_windowed(N, d, bs, W; stride=W, pad=0, reps=100, T=Float64)
    shape = (N, d, bs)
    # init variables
    Q = randn(T, shape)
    K = randn(T, shape)
    V = randn(T, shape)
    O1 = similar(Q, shape)
    O2 = similar(Q, shape)

    kws = Dict(:stride=>stride, :pad=>pad)
    args = Q, K, V, W

    # warmup
    O1 = windowed_dpa(args...; kws...) |> first
    O2 = windowed_fa(args...; kws...) |> first
    @test O1 ≈ O2

    time_dpa = 0.0
    time_fa  = 0.0
    for i=1:reps
        time_dpa += @elapsed windowed_dpa(args...; kws...)
        time_fa  += @elapsed  windowed_fa(args...; kws...)
    end

    return time_dpa/reps, time_fa/reps
end

function time_circulant(N, d, bs, W; reps=100, T=Float64)
    shape = (N, d, bs)
    # init variables
    Q = randn(T, shape)
    K = randn(T, shape)
    V = randn(T, shape)
    O1 = similar(Q, shape)
    O2 = similar(Q, shape)
    P = similar(Q, W, N, bs)
    l = similar(Q, N, 1, bs)
    m = similar(Q, N, 1, bs)

    # warmup
    O1 = circulant_dpa!(O1, P, Q, K, V, W) |> first
    O2 = circulant_fa!(O2, l, m, Q, K, V, W) |> first
    @test O1 ≈ O2

    time_dpa = 0.0
    time_fa  = 0.0
    for i=1:reps
        time_dpa += @elapsed circulant_dpa!(O1, P, Q, K, V, W)
        time_fa  += @elapsed circulant_fa!(O2, l, m, Q, K, V, W)
    end

    return time_dpa/reps, time_fa/reps
end

function runcompare(; N_range=2 .^ (8:10), d_range=(32,), bs_range=(1,), windowsize=128, reps=100)
    @printf("%5s %5s %5s %10s %10s %10s %10s %10s %10s %10s %10s\n", 
            "N", "d", "bs", 
            "dense_dpa", "dense_fa", 
            "block_dpa", "block_fa", 
            "wind_dpa", "wind_fa", 
            "circ_dpa", "circ_fa")

    for N in N_range, d in d_range, bs in bs_range
        tdense = time_dense(N, d, bs; reps=reps)
        tblock = time_windowed(N, d, bs, windowsize; reps=reps)
        twind  = time_windowed(N, d, bs, windowsize; stride=16, reps=reps)
        tcirc  = time_circulant(N, d, bs, windowsize + 1; reps=reps)

        @printf("%5d %5d %5d %10f %10f %10f %10f %10f %10f %10f %10f\n", 
                N, d, bs, tdense..., tblock..., twind..., tcirc...)
    end
end

function runwindow(window_range=2 .^(4:8); stride=8, N=4096, d=32, bs=1, reps=100)
    @printf("%5s %5s %5s %5s %10s %10s\n", 
            "N", "d", "bs", "W",
            "wind_dpa", "wind_fa")

    for W in window_range
        twind  = time_windowed(N, d, bs, W; stride=stride, reps=reps)

        @printf("%5d %5d %5d %5d %10f %10f\n", 
                N, d, bs, W, twind...)
    end
end

function runcirculant(window_range=2 .^(4:8); stride=8, N=4096, d=32, bs=1, reps=100)
    @printf("%5s %5s %5s %5s %10s %10s\n", 
            "N", "d", "bs", "W",
            "circ_dpa", "circ_fa")

    for W in window_range
        tcirc = time_circulant(N, d, bs, W; reps=reps)

        @printf("%5d %5d %5d %5d %10f %10f\n", 
                N, d, bs, W, tcirc...)
    end
end
