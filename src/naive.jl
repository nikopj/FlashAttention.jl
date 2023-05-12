function dense_dpa(Q::AbstractArray{T, 3}, K::AbstractArray{T, 3}, V::AbstractArray{T, 3}) where T
    d = size(Q, 2)
    S = (Q ⊠ batched_transpose(K)) ./ T(sqrt(d))  # similarity matrix
    P = NNlib.softmax(S, dims=2)                  # adjacency matrix
    O = P ⊠ V                                     # output
    return O, P
end

function dense_dpa(q::AbstractArray{T, N}, k::AbstractArray{T, N}, v::AbstractArray{T, N}) where {T, N}
    dqk = size(q, N-1)
    dvo = size(v, N-1)
    bs  = size(q, N)

    Q = reshape(q, :, dqk, bs)
    K = reshape(k, :, dqk, bs)
    V = reshape(v, :, dvo, bs)

    O, P = dense_dpa(Q, K, V)

    y = reshape(O, size(q)[1:N-2]..., dvo, bs)
    return y, P
end

function dense_dpa_backward(
    Q::AbstractArray{T, 3}, 
    K::AbstractArray{T, 3}, 
    V::AbstractArray{T, 3}, 
    P::AbstractArray{T, 3},
    dO::AbstractArray{T, 3}) where T

    dV = batched_transpose(dP) ⊠ dO
    dP = dO ⊠ batched_transpose(V)
    dS = P .* (dP .- sum(P .* dP, dims=2))
    dQ = (dS ⊠ K) ./ T(sqrt(d))
    dK = (batched_transpose(dS) ⊠ Q) ./ T(sqrt(d))
    return dQ, dK, dV
end

function window(x::AbstractArray{T, N}, windowsize; stride=windowsize, pad=0) where {T, N}
    # input shape: length, channels, batchsize
    # output shape: windowsize, channels, nwindows, batchsize
    d = size(x, N-1)
    X = NNlib.unfold(x, (ntuple(i->windowsize, N-2)..., d, 1); stride=stride, pad=pad)
    X = permutedims(X, (2, 1, 3))
    X = reshape(X, windowsize^(N-2), d, :, size(x, N))
    return X
end

function unwindow(X::AbstractArray{T, N2}, outputsize::NTuple{N}, windowsize; stride=windowsize, pad=0) where {T, N, N2}
    # input shape: windowsize, channels, nwindows, batchsize
    # output shape: length, channels, batchsize
    d = size(X, N2-2)
    X = reshape(X, windowsize^(N-2)*d, :, size(X, N2))
    X = permutedims(X, (2, 1, 3))
    x = NNlib.fold(X, outputsize, (ntuple(i->windowsize, N-2)..., d, 1); stride=stride, pad=pad)
    return x
end

function windowed_dpa(q::AbstractArray{T, N}, k::AbstractArray{T, N}, v::AbstractArray{T, N}, windowsize; kws...) where {T, N}
    qw = window(q, windowsize; kws...)
    kw = window(k, windowsize; kws...)
    vw = window(v, windowsize; kws...)

    yw, Pw = dense_dpa(
        reshape(qw, size(qw, 1), size(qw, 2), :),
        reshape(kw, size(kw, 1), size(kw, 2), :),
        reshape(vw, size(vw, 1), size(vw, 2), :))

    yw = reshape(yw, size(yw, 1), size(yw, 2), size(qw, 3), :)
    szy = (size(q)[1:N-2]..., size(v, N-1), size(q, N))

    divisor = unwindow(window(ones_like(v, szy), windowsize; kws...),
        szy, windowsize; kws...)

    y = unwindow(yw, szy, windowsize; kws...) ./ divisor
    P = reshape(Pw, size(Pw, 1), size(Pw, 2), :, size(q, N))
    return y, P
end
    
function circulant_dpa(Q::AbstractArray{T, 3}, K::AbstractArray{T, 3}, V::AbstractArray{T, 3}, W::Int) where T
    N, d, batchsize = size(Q)
    τ = one(T) / T(sqrt(d))
    S = similar(Q, N, W, batchsize)
    O = similar(Q, N, size(V, 2), batchsize)

    # form similarity matrix
    @threads for idx in CartesianIndices((N, W, batchsize))
        ii, ww, bb = idx.I
        n = (ii-1)*W + ww
        jj = cartesian_circulant(n, N, W)[1]
        S[ii, ww, bb] = τ * sum(Q[ii, :, bb].*K[jj, :, bb])
    end

    # normalize
    P = softmax(S, dims=2)

    # CHANGE TO BLAS SPARSE-MAT x DENSE-VEC
    # attend
    @threads for idx in CartesianIndices((N, d, batchsize))
        ii, dd, bb = idx.I
        t = zero(T)
        for ww=1:W
            n = (ii-1)*W + ww
            jj = cartesian_circulant(n, N, W)[1]
            t += P[ii, ww, bb] * V[jj, dd, bb]
        end
        O[ii, dd, bb] = t
    end

    return O, P
end
