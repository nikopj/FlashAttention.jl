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
    
function circulant_dpa(q::AbstractArray{T, 3}, k::AbstractArray{T, 3}, v::AbstractArray{T, 3}, windowsize; kws...) where {T, N}
    @assert size(q, 1) == size(k, 1) && size(k, 1) == size(v, 1)

    S = circulant(size(q, 1), windowsize, T)

    for n in 1:size(S, 1)*size(S, 2)
        i, j = cartesian_circulant(n, size(S, 1), windowsize)
        # 1xdxb X 1xdxb -> 1x1xb

        # we want a new struct that holds a batched circulant matrix.
        # ex. 
        # struct
        #   nzval::AbstractArray{T, 3}
        #   dims::
        # end
    end
end
