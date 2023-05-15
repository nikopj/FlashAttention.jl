function dense_fa(q::AbstractArray{T, D}, k::AbstractArray{T, D}, v::AbstractArray{T, D}) where {T, D}
    d  = size(q, D-1)
    dv = size(v, D-1)
    batchsize = size(q, D)

    Q = reshape(q, :, d, batchsize)
    K = reshape(k, :, d, batchsize)
    V = reshape(v, :, dv,  batchsize)

    N = size(Q, 1)
    O = similar(Q)
    l = similar(Q, N, 1, batchsize)
    m = similar(Q, N, 1, batchsize)

    dense_fa!(O, l, m, Q, K, V)

    y = reshape(O, size(q)[1:D-2]..., dv, :)
    return y, l, m
end

function dense_fa!(
        O::AbstractArray{T, 3}, 
        l::AbstractArray{T, 3}, 
        m::AbstractArray{T, 3}, 
        Q::AbstractArray{T, 3}, 
        K::AbstractArray{T, 3}, 
        V::AbstractArray{T, 3}) where {T}
    M = 32_000 # ≈ cache size
    N, d, batchsize = size(Q)

    # row/column block-length
    # Bc = clamp(cld(M, 4d), 1, N)
    # Br = clamp(min(d, cld(M, 4d)), 1, N)
    Bc = clamp(cld(M, d), 1, N)
    Br = clamp(min(d, cld(M, d)), 1, N)
    Bb = 1

    # num row/column blocks
    Tb = cld(batchsize, Bb)
    Tr = cld(N, Br)
    Tc = cld(N, Bc)

    τ = one(T)/T(sqrt(d))

    @threads for c in CartesianIndices((Tb, Tr))
        b, i = c.I
        start_bdx = (b-1)*Bb + 1
        end_bdx = min(batchsize, b*Bb)

        start_idx = (i-1)*Br + 1
        end_idx = min(N, i*Br)

        @views Qi = Q[start_idx:end_idx, :, start_bdx:end_bdx]
        @views Oi = O[start_idx:end_idx, :, start_bdx:end_bdx]
        @views li = l[start_idx:end_idx, :, start_bdx:end_bdx]
        @views mi = m[start_idx:end_idx, :, start_bdx:end_bdx]

        fill!(Oi, zero(T))
        fill!(li, zero(T))
        fill!(mi, T(-Inf))

        Oi_new = similar(Oi)
        mij = similar(mi)
        lij = similar(li)
        eij = similar(li)
        ei  = similar(li)
        mi_new = similar(mi)
        li_new = similar(li)

        for j=1:Tc
            start_jdx = (j-1)*Bc + 1
            end_jdx = min(N, j*Bc)
            @views Kj = K[start_jdx:end_jdx, :, start_bdx:end_bdx]
            @views Vj = V[start_jdx:end_jdx, :, start_bdx:end_bdx]
            Pij = similar(Qi, size(Qi, 1), size(Kj, 1), size(Qi, 3))
             
            batched_mul!(Pij, Qi, batched_transpose(Kj), τ)
            maximum!(mij, Pij)
            @. Pij = exp(Pij - mij)                          
            sum!(lij, Pij)

            @. mi_new = max(mi, mij)
            @. ei  = exp(mi - mi_new)
            @. eij = exp(mij - mi_new)
            @. li_new = ei*li + eij*lij

            # write back to memory
            batched_mul!(Oi_new, Pij, Vj)
            @. Oi = (li*ei*Oi + eij*Oi_new) / li_new
            li .= li_new
            mi .= mi_new
        end

        # these writes are commented out because they're already happening 
        # via Oi, li, mi being views of O, l, m.
         
        # O[start_idx:end_idx, :, start_bdx:end_bdx] = Oi 
        # l[start_idx:end_idx, :, start_bdx:end_bdx] = li 
        # m[start_idx:end_idx, :, start_bdx:end_bdx] = mi 
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

    M  = 1024000 # SRAM size
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

