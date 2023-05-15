using FlashAttention
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
    b = @test O1 ≈ O2
    println(b)

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
    b = @test O1 ≈ O2
    println(b)

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
    b = @test O1 ≈ O2
    println(b)

    time_dpa = 0.0
    time_fa  = 0.0
    for i=1:reps
        time_dpa += @elapsed circulant_dpa!(O1, P, Q, K, V, W)
        time_fa  += @elapsed circulant_fa!(O2, l, m, Q, K, V, W)
    end

    return time_dpa/reps, time_fa/reps
end
