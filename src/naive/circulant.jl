@inline function circulant_dpa(Q, K, V, W)
    N, d, bs = size(Q)
    O = similar(Q, N, size(V, 2), batchsize)
    P = similar(Q, W, N, batchsize)
    return circulant_dpa!(O, P, Q, K, V, W)
end

function circulant_dpa!(
        O::AbstractArray{T, 3}, 
        P::AbstractArray{T, 3}, 
        Q::AbstractArray{T, 3}, 
        K::AbstractArray{T, 3}, 
        V::AbstractArray{T, 3}, 
        W::Int) where T
    N, d, batchsize = size(Q)
    τ = one(T) / T(sqrt(d))

    # form similarity matrix's nzvals
    @threads for idx in CartesianIndices((N, W, batchsize))
        ii, ww, bb = idx.I
        n = (ii-1)*W + ww
        jj = cartesian_circulant(n, N, W)[1]
        P[ww, ii, bb] = τ * sum(Q[ii, :, bb].*K[jj, :, bb])
    end

    # normalize and convert to sparse representation
    softmax!(P, dims=1)
    P = batch_circulant(P) |> transpose
    
    # concat batch-elements in sequence direction
    bV = vcat([V[:,:,b] for b=1:batchsize]...)
    bO = P * bV

    O  = cat([bO[(b-1)*N+1:b*N, :] for b=1:batchsize]...; dims=3)
    return O, P
end
