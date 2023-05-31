using CUDA
using FlashAttention
using FlashAttention: fused_softmax!, sm_naive!
using NNlib, NNlibCUDA
using Printf
using Test

function time_vec_softmax(N; reps=100, T=Float32)
    # init variables
    U1 = CUDA.rand(T, N)
    U2 = CUDA.rand(T, N)
    U3 = CUDA.rand(T, N)
    V  = CUDA.rand(T, N)

    # warmup
    U1 = sm_naive!(U1, V) 
    U2 = fused_softmax!(U2, V)
    U3 = NNlib.softmax!(U3, V)
    
    @test U1 ≈ U3 
    @test U2 ≈ U3 

    time_naive = 0.0
    time_fused = 0.0
    time_nnlib = 0.0
    for i=1:reps
        time_naive += CUDA.@elapsed CUDA.@sync sm_naive!(U1, V)
        time_fused += CUDA.@elapsed CUDA.@sync fused_softmax!(U2, V)
        time_nnlib += CUDA.@elapsed CUDA.@sync NNlib.softmax!(U3, V)
    end

    return time_naive/reps, time_fused/reps, time_nnlib/reps
end

function time_col_softmax(M, N; reps=100, T=Float32)
    # init variables
    U1 = CUDA.rand(T, M, N)
    U2 = CUDA.rand(T, M, N)
    U3 = CUDA.rand(T, M, N)
    V  = CUDA.rand(T, M, N)

    # warmup
    U1 = sm_naive!(U1, V) 
    U2 = fused_softmax!(U2, V)
    U3 = NNlib.softmax!(U3, V)
    
    @test U1 ≈ U3 
    @test U2 ≈ U3 

    time_naive = 0.0
    time_fused = 0.0
    time_nnlib = 0.0
    for i=1:reps
        time_naive += CUDA.@elapsed CUDA.@sync sm_naive!(U1, V)
        time_fused += CUDA.@elapsed CUDA.@sync fused_softmax!(U2, V)
        time_nnlib += CUDA.@elapsed CUDA.@sync NNlib.softmax!(U3, V)
    end

    return time_naive/reps, time_fused/reps, time_nnlib/reps
end

function run_vec_softmax(; N_range=2 .^ (10:16), reps=100)
    @printf("%10s %10s %10s %10s\n", "N", "naive", "fused", "nnlib")

    for N in N_range
        t = time_vec_softmax(N; reps=reps)
        @printf("%10d %10f %10f %10f\n", N, t...)
    end
end

function run_col_softmax(; M_range=(1024,), N_range=2 .^ (10:16), reps=100)
    @printf("%10s %10s %10s %10s %10s\n", "M", "N", "naive", "fused", "nnlib")

    for M in M_range, N in N_range
        t = time_col_softmax(M, N; reps=reps)
        @printf("%10d %10d %10f %10f %10f\n", M, N, t...)
    end
end
