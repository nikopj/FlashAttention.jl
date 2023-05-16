@inline function dense_dpa(Q, K, V)
    N, d, batchsize = size(Q) 
    O = similar(Q)
    P = similar(Q, N, N, bs)
    return dense_dpa!(O, P, Q, K, V)
end

function dense_dpa!(
        O::AbstractArray{T, 3}, 
        P::AbstractArray{T, 3}, 
        Q::AbstractArray{T, 3}, 
        K::AbstractArray{T, 3}, 
        V::AbstractArray{T, 3}) where T
    batched_mul!(P, Q, batched_transpose(K), one(T) / sqrt(size(Q, 2)), zero(T)) 
    NNlib.softmax!(P, dims=2)           
    batched_mul!(O, P, V)
    return O, P
end

function dense_dpa(q::AbstractArray{T, N}, k::AbstractArray{T, N}, v::AbstractArray{T, N}) where {T, N}
    dqk = size(q, N-1)
    dvo = size(v, N-1)
    bs  = size(q, N)

    Q = reshape(q, :, dqk, bs)
    K = reshape(k, :, dqk, bs)
    V = reshape(v, :, dvo, bs)
    O = similar(Q, size(Q, 1), dvo, bs)
    P = similar(Q, size(Q, 1), size(Q, 1), bs)

    dense_dpa!(O, P, Q, K, V)

    y = reshape(O, size(q)[1:N-2]..., dvo, bs)
    return y, P
end

function dense_dpa_backward!(
    dQ::AbstractArray{T, 3}, 
    dK::AbstractArray{T, 3}, 
    dV::AbstractArray{T, 3}, 
    Q::AbstractArray{T, 3}, 
    K::AbstractArray{T, 3}, 
    V::AbstractArray{T, 3}, 
    P::AbstractArray{T, 3},
    dO::AbstractArray{T, 3}) where T

    τ = one(T) / T(sqrt(d))
    dP = dO ⊠ batched_transpose(V)
    s  = sum(P .* dP, dims=2)
    dS = @. P * (dP - s)

    batched_mul!(dQ, dS, K, τ)
    batched_mul!(KQ, batched_transpose(dS), Q, τ)
    batched_mul!(dV, batched_transpose(P), dO)
    return dQ, dK, dV
end
