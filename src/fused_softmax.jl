fused_softmax(S; dims=1) = fused_softmax!(similar(S), S; dims=dims)
fused_softmax!(S; dims=1) = fused_softmax!(S, S; dims=dims)

function fused_softmax!(P::AbstractMatrix{T}, S::AbstractMatrix{T}; dims=1) where T
    P = reshape(P, size(P)..., 1)
    S = reshape(S, size(S)..., 1)
    fused_softmax!(P, S; dims=dims)
    return reshape(P, size(P, 1), size(P, 2))
end

function fused_softmax!(P::AbstractArray{T, 3}, S::AbstractArray{T, 3}; dims=1) where T
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
