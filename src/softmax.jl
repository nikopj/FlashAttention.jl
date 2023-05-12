softmax(S; dims=1) = softmax!(similar(S), S; dims=dims)
softmax!(S; dims=1) = softmax!(S, S; dims=dims)

function softmax!(P::AbstractMatrix{T}, S::AbstractMatrix{T}; dims=1) where T
    P = reshape(P, size(P)..., 1)
    S = reshape(S, size(S)..., 1)
    softmax!(P, S; dims=dims)
    return P[:,:,1]
end

function softmax!(P::AbstractArray{T, 3}, S::AbstractArray{T, 3}; dims=1) where T
    @assert dims in (1, 2) "only softmax in dims 1 or 2 supported"
    dims == 1 ? col_softmax!(P, S) : row_softmax!(P, S)
    return P
end

function row_softmax!(P::AbstractArray{T, 3}, S::AbstractArray{T, 3}) where T
    @threads for c in CartesianIndices((size(S, 1), size(S, 3)))
        i, b = c.I
        s = S[i, :, b] # load row
        s_minus_max = s .- maximum(s)
        numerator   = exp.(s_minus_max)
        denominator = sum(numerator)
        P[i, :, b] = numerator ./ denominator
    end
    return P
end

function col_softmax!(P::AbstractArray{T, 3}, S::AbstractArray{T, 3}) where T
    @threads for c in CartesianIndices((size(S, 2), size(S, 3)))
        j, b = c.I
        s = S[:, j, b] # load row
        s_minus_max = s .- maximum(s)
        numerator   = exp.(s_minus_max)
        denominator = sum(numerator)
        P[:, j, b] = numerator ./ denominator
    end
    return P
end
