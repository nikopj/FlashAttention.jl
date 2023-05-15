@inline function circulant_fa(Q, K, V, W)
    N, d, batchsize = size(Q)
    O = similar(Q)
    l = similar(Q, N, 1, batchsize)
    m = similar(Q, N, 1, batchsize)
    return circulant_fa!(O, l, m, Q, K, V)
end

function circulant_fa!(
        O::AbstractArray{T, 3}, 
        l::AbstractArray{T, 3}, 
        m::AbstractArray{T, 3}, 
        Q::AbstractArray{T, 3}, 
        K::AbstractArray{T, 3}, 
        V::AbstractArray{T, 3},
        W::Int) where {T}
    # W is the windowsize 
    M = 32_000 # SRAM size
    N, d, batchsize = size(Q)

    # row/column block-length
    # Bw = clamp(cld(M, 4d), 1, N)
    # Br = clamp(min(d, cld(M, 4d)), 1, N)
    Bw = clamp(cld(M, d), 1, W)
    Br = clamp(min(d, cld(M, d)), 1, N)
    Bb = 1

    # num row/column blocks
    Tb = cld(batchsize, Bb)
    Tr = cld(N, Br)
    Tw = cld(W, Bw)

    τ = one(T)/T(sqrt(d))

    @threads for c in CartesianIndices((Tb, Tr))
        b, i = c.I
        start_bdx = (b-1)*Bb + 1
        end_bdx = min(batchsize, b*Bb)

        start_idx = (i-1)*Br + 1
        end_idx = min(N, i*Br)

        row_range   = start_idx:end_idx
        batch_range = start_bdx:end_bdx

        @views Oi = O[row_range, :, batch_range]
        @views li = l[row_range, :, batch_range]
        @views mi = m[row_range, :, batch_range]

        fill!(Oi, zero(T))
        fill!(li, zero(T))
        fill!(mi, T(-Inf))

        miw = similar(mi)
        liw = similar(li)
        eiw = similar(li)
        ei  = similar(li)
        mi_new = similar(mi)
        li_new = similar(li)

        for w=1:Tw
            start_wdx = (w-1)*Bw + 1
            end_wdx = min(W, w*Bw)
            window_range = start_wdx:end_wdx
            Piw = similar(Oi, size(Oi, 1), length(window_range), size(Oi, 3))
             
            # form Piwk
            for ii=1:length(row_range), ww=1:length(window_range), bb=1:length(batch_range)
                iii = row_range[ii]
                bbb = batch_range[bb]
                www = window_range[ww]
                nnn = (iii-1)*W + www
                jjj = cartesian_circulant(nnn, N, W)[1]
                t = zero(T)
                for kk=1:d
                    t += Q[iii, kk, bbb]*K[jjj, kk, bbb]
                end
                Piw[ii, ww, bb] = τ*t
            end
            maximum!(miw, Piw)
            @. Piw = exp(Piw - miw)                          
            sum!(liw, Piw)

            @. mi_new = max(mi, miw)
            @. ei  = exp(mi - mi_new)
            @. eiw = exp(miw - mi_new)
            @. li_new = ei*li + eiw*liw

            # Oi_new = Piw * V, with circulant indexing
            for ii=1:length(row_range), dd=1:d, bb=1:length(batch_range)
                iii = row_range[ii]
                bbb = batch_range[bb]
                t = zero(T) # Oi_new
                for ww=1:length(window_range)
                    www = window_range[ww]
                    nnn = (iii-1)*W + www
                    jjj = cartesian_circulant(nnn, N, W)[1]
                    t += Piw[ii, ww, bb]*V[jjj, dd, bbb]
                end
                #Oi_new[ii, dd, bb] = t
                Oi[ii, dd, bb] = (li[ii, 1, bb]*ei[ii, 1, bb]*Oi[ii, dd, bb] + eiw[ii, 1, bb]*t) / li_new[ii, 1, bb]
            end

            # write back to memory
            #@. Oi = (li*exp(mi - mi_new)*Oi + exp(miw - mi_new)*Oi_new) / li_new
            li .= li_new
            mi .= mi_new
        end

        # these writes are commented out because they're already happening 
        # via Oi, li, mi being views of O, l, m.
    
        # O[row_range, :, batch_range] = Oi 
        # l[row_range, :, batch_range] = li 
        # m[row_range, :, batch_range] = mi 
    end
    return O, l, m
end
