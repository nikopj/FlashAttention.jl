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

function circulant_fa(Q::AbstractArray{T, 3}, K::AbstractArray{T, 3}, V::AbstractArray{T, 3}, W::Int) where {T}
    # W is the windowsize 
    M = 32000 # SRAM size
    N, d, batchsize = size(Q)

    # row/column block-length
    Bw = clamp(ceil(Int, M/d), 1, W)
    #Br = clamp(min(d, ceil(Int, M/d)), 1, N)
    Br = clamp(ceil(Int, M/8d), 1, N)
    Bb = 1

    # num row/column blocks
    Tb = cld(batchsize, Bb)
    Tr = cld(N, Br)
    Tw = cld(W, Bw)

    # initialize variables
    O = similar(V)
    l = similar(Q, N, 1, batchsize)
    m = similar(Q, N, 1, batchsize)
    τ = one(T)/T(sqrt(d))

    fill!(O, zero(T))
    fill!(l, zero(T))
    fill!(m, T(-Inf))

    @threads for c in CartesianIndices((Tb, Tr))
        b, i = c.I
        start_bdx = (b-1)*Bb + 1
        end_bdx = min(batchsize, b*Bb)

        start_idx = (i-1)*Br + 1
        end_idx = min(N, i*Br)

        row_range   = start_idx:end_idx
        batch_range = start_bdx:end_bdx

        Oi = O[row_range, :, batch_range]
        li = l[row_range, :, batch_range]
        mi = m[row_range, :, batch_range]

        Oi_new = similar(Oi)
        miw = similar(mi)
        liw = similar(li)
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
                    t += τ*Q[iii, kk, bbb]*K[jjj, kk, bbb]
                end
                Piw[ii, ww, bb] = t
            end
            maximum!(miw, Piw)
            @. Piw = exp(Piw - miw)                          
            sum!(liw, Piw)

            @. mi_new = max(mi, miw)
            @. li_new = exp(mi - mi_new)*li + exp(miw - mi_new)*liw

            # Oi_new = Piw * V, with circulant indexing
            for ii=1:length(row_range), dd=1:d, bb=1:length(batch_range)
                iii = row_range[ii]
                bbb = batch_range[bb]
                t = zero(T)
                for ww=1:length(window_range)
                    www = window_range[ww]
                    nnn = (iii-1)*W + www
                    jjj = cartesian_circulant(nnn, N, W)[1]
                    t += Piw[ii, ww, bb]*V[jjj, dd, bbb]
                end
                Oi_new[ii, dd, bb] = t
            end

            # write back to memory
            @. Oi = (li*exp(mi - mi_new)*Oi + exp(miw - mi_new)*Oi_new) / li_new
            li .= li_new
            mi .= mi_new
        end

        O[row_range, :, batch_range] = Oi 
        l[row_range, :, batch_range] = li 
        m[row_range, :, batch_range] = mi 
    end
    return O, l, m
end
