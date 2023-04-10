function dense_fa(q::AbstractArray{T, D}, k::AbstractArray{T, D}, v::AbstractArray{T, D}) where {T, D}
    d  = size(q, D-1)
    dv = size(v, D-1)
    Q = reshape(q, :, d, batchsize)
    K = reshape(k, :, d, batchsize)
    V = reshape(v, :, dv,  batchsize)

    O, l, m = dense_fa(Q, K, V)

    y = reshape(O, size(q)[1:D-2]..., dv, :)
    return y, l, m
end

function dense_fa(Q::AbstractArray{T, D}, K::AbstractArray{T, D}, V::AbstractArray{T, D}) where {T, D}
    const M  = 1024000 # SRAM size
    N, batchsize = size(q, 1), size(q, D)
    d  = size(q, D-1)

    # row/column block-length
    Br = min(d, cld(M, 4*d*batchsize))
    Bc = cld(M, 4*d*batchsize)

    # num row/column blocks
    Tr = cld(N, Br)
    Tc = cld(N, Bc)

    # initialize variables
    O = similar(V)
    l = similar(Q, N, 1, batchsize)
    m = similar(Q, N, 1, batchsize)

    fill!(O, zero(T))
    fill!(l, zero(T))
    fill!(m, T(-Inf))

    @threads for i=1:Tr
        start_idx = (i-1)*Br + 1
        end_idx = min(N, i*Br)
        Qi = @view Q[start_idx:end_idx, :, :]
        Oi = O[start_idx:end_idx, :, :]
        li = l[start_idx:end_idx, :, :]
        mi = m[start_idx:end_idx, :, :]

        for j=1:Tc
            start_jdx = (j-1)*Bc + 1
            end_jdx = min(N, j*Bc)
            Kj = @view K[start_jdx:end_jdx, :, :]
            Vj = @view V[start_jdx:end_jdx, :, :]

            Sij = (Qi ⊠ batched_transpose(Kj)) ./ T(sqrt(d))  # similarity matrix
            mij = maximum(Sij, dims=2)                          

            Pij = @. exp(Sij - mij)                           # adjacency matrix
            lij = sum(Pij, dims=2)

            mi_new = max.(mi, mij)
            li_new = @. exp(mi - mi_new)*li + exp(mij - mi_new)*lij

            # write back to memory
            @. Oi = (li*exp(mi - mi_new)*Oi + exp(mij - mi_new)*$batched_mul(Pij, Vj)) / li_new
            li .= li_new
            mi .= mi_new
        end

        O[start_idx:end_idx, :, :] = Oi 
        l[start_idx:end_idx, :, :] = li 
        m[start_idx:end_idx, :, :] = mi 
    end
    return O, l, m
end

function dense_fa_backward(
    Q::AbstractArray{T, 3}, 
    K::AbstractArray{T, 3}, 
    V::AbstractArray{T, 3}, 
    O::AbstractArray{T, 3}, 
    dO::AbstractArray{T, 3},
    l::AbstractArray{T, 3},
    m::AbstractArray{T, 3}) where T

    const M  = 1024000 # SRAM size
    N, batchsize = size(q, 1), size(q, D)
    d  = size(q, D-1)

    # row/column block-length
    Br = min(d, cld(M, 4*d*batchsize))
    Bc = cld(M, 4*d*batchsize)

    # num row/column blocks
    Tr = cld(N, Br)
    Tc = cld(N, Bc)

    # initialize variables
    dQ = similar(Q)
    dK = similar(K)
    dV = similar(V)

    fill!(O, zero(T))

    @threads for i=1:Tr
        start_idx = (i-1)*Br + 1
        end_idx = min(N, i*Br)
        Qi  =  Q[start_idx:end_idx, :, :]
        Oi  =  O[start_idx:end_idx, :, :]
        dQi = dQ[start_idx:end_idx, :, :]
        dOi = dO[start_idx:end_idx, :, :]
        li  =  l[start_idx:end_idx, :, :]
        mi  =  m[start_idx:end_idx, :, :]

        for j=1:Tc
            start_jdx = (j-1)*Bc + 1
            end_jdx = min(N, j*Bc)
            Kj  =  K[start_jdx:end_jdx, :, :]
            Vj  =  V[start_jdx:end_jdx, :, :]
            dKj = @view dK[start_jdx:end_jdx, :, :]
            dVj = @view dV[start_jdx:end_jdx, :, :]

            Sij = (Qi ⊠ batched_transpose(Kj)) ./ T(sqrt(d))  
            Pij = @. exp(Sij - mi) / li        
            
            dVj += batched_transpose(Pij) ⊠ dOi
            dPij = dOi ⊠ batched_transpose(dVj)

            Di   = sum(dOi .* Oi, dims=2)
            dSij = @. Pij * (dPij - Di)

            dQi += (dSij ⊠ Kj) ./ T(sqrt(d))
            dKj += (batched_transpose(dSij) ⊠ Qi) ./ T(sqrt(d))
        end

        dQ[start_idx:end_idx, :, :] = dQi 
    end
    return dQ, dK, dV
end

