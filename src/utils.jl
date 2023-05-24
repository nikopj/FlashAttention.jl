# m: filter index, 
# s: shift
# M: filter-length
circshift_index(m, s, M) = mod(m - 1 - s, M) + 1

function cartesian_circulant(n, N, M)
    # filter size must be odd
    p = (M-1) ÷ 2
    j = cld(n, M) # col num
    m = mod(n-1, M) + 1
    if j <= p
        m = circshift_index(m, j - p - 1, M) elseif j > N-p 
        m = circshift_index(m, p - N + j, M)
    end
    i = mod((m-1) + (j-1) - p, N) + 1
    return i, j
end

function circulant(N::Int, M::Int, Tv=Float64, Ti=Int64) 
    rowval = (Ti∘first∘cartesian_circulant).(1:N*M, N, M) 
    colptr = 1 .+ M .* collect(0:N) .|> Ti
    return SparseMatrixCSC{Tv, Ti}(N, N, colptr, rowval, ones(Tv, N*M))
end

function circulant(V::AbstractMatrix{Tv}, Ti=Int64) where Tv
    M, N = size(V)
    rowval = (Ti∘first∘cartesian_circulant).(1:N*M, N, M) 
    colptr = 1 .+ M .* collect(0:N) .|> Ti
    return SparseMatrixCSC{Tv, Ti}(N, N, colptr, rowval, reshape(V, :))
end

function batch_circulant(bV::AbstractArray{Tv, 3}, Ti=Int64) where Tv
    return blockdiag([circulant(bV[:,:,b]) for b=1:size(bV, 3)]...)
end

function window(x::AbstractArray{T, N}, windowsize; stride=windowsize, pad=(windowsize-1)÷2) where {T, N}
    # input shape: length, channels, batchsize
    # output shape: windowsize, channels, nwindows, batchsize
    d = size(x, N-1)
    X = NNlib.unfold(x, (ntuple(i->windowsize, N-2)..., d, 1); stride=stride, pad=pad)
    X = permutedims(X, (2, 1, 3))
    X = reshape(X, windowsize^(N-2), d, :, size(x, N))
    return X
end

function unwindow(X::AbstractArray{T, N2}, outputsize::NTuple{N}, windowsize; stride=windowsize, pad=(windowsize-1)÷2) where {T, N, N2}
    # input shape: windowsize, channels, nwindows, batchsize
    # output shape: length, channels, batchsize
    d = size(X, N2-2)
    X = reshape(X, windowsize^(N-2)*d, :, size(X, N2))
    X = permutedims(X, (2, 1, 3))
    x = NNlib.fold(X, outputsize, (ntuple(i->windowsize, N-2)..., d, 1); stride=stride, pad=pad)
    return x
end

