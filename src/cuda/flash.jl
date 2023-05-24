function dense_fa_kernel!(
        O::CuDeviceArray{T, 3}, 
        l::CuDeviceArray{T, 3},
        m::CuDeviceArray{T, 3},
        Q::CuDeviceArray{T, 3}, 
        K::CuDeviceArray{T, 3}, 
        V::CuDeviceArray{T, 3}, 
        N::Int32, d::Int32, B::Int32) where T
    # ASSUME THAT Br == Bc
    ti, tj, tb = threadIdx().x, threadIdx().y, threadIdx().z
    Br, Bc, Bb = blockDim().x,  blockDim().y,  blockDim().z

    row = (blockIdx().x - 1i32)*Br + ti
    bat = (blockIdx().z - 1i32)*Bb + tb

    # load Oi, Qi, mi, li into shared memory
    Qi = CuDynamicSharedArray(T, (Br, d, Bb))
    Oi = CuDynamicSharedArray(T, (Br, d, Bb), sizeof(T)*Br*d*Bb)
    li = CuDynamicSharedArray(T, (Br, 1,  Bb), sizeof(T)*2*Br*d*Bb)
    mi = CuDynamicSharedArray(T, (Br, 1,  Bb), sizeof(T)*Bb*Br*(2*d + 1))

    @inbounds if tj == 1i32
        for k=1i32:d
            Qi[ti, k, tb] = (row <= N && bat <= B) ? Q[row, k, bat] : zero(T)
            Oi[ti, k, tb] = zero(T)
        end
        li[ti, 1, tb] = zero(T)
        mi[ti, 1, tb] = T(-Inf)
    end

    # allocate Kj, Vj
    Kj  = CuDynamicSharedArray(T, (Bc, d, Bb), sizeof(T)*2*Bb*Br*(d + 1))
    Vj  = CuDynamicSharedArray(T, (Bc, d, Bb), sizeof(T)*Bb*(2*Br*(d + 1) + Bc*d))
    Pij = CuDynamicSharedArray(T, (Br, Bc, Bb), sizeof(T)*Bb*(2*Br*(d + 1) + 2*Bc*d))

    # set DPA scale factor
    τ = T(1i32 / sqrt(d))

    @inbounds for j=1i32:cld(N, Bc)
        col = (j-1i32)*Bc + tj

        # load Kj, Vj into shared memory
        if ti == 1i32
            for k=1i32:d
                Kj[tj, k, tb] = (col <= N && bat <= B) ? K[col, k, bat] : zero(T)
                Vj[tj, k, tb] = (col <= N && bat <= B) ? V[col, k, bat] : zero(T)
            end
        end
        # make sure all data is loaded
        sync_threads()

        # Pij = τ * Qi ⊠ Kj'
        Pval = zero(T)
        for k=1i32:d
            Pval += τ * Qi[ti, k, tb] * Kj[tj, k, tb]
        end
        Pij[ti, tj, tb] = Pval
        sync_threads()

        # compute local row-max
        mij = Pij[ti, 1, tb]
        for k=2i32:Bc
            mij = max(mij, Pij[ti, k, tb])
        end
        Pij[ti, tj, tb] = exp(Pij[ti, tj, tb] - mij)
        sync_threads()

        # compute local row-sum
        lij = zero(T)
        for k=1i32:Bc
            lij += Pij[ti, k, tb]
        end

        # compute updated local row-max, local row-sum
        idx1 = CartesianIndex((ti, 1, tb))
        mi_new = max(mi[idx1], mij)
        expmi  = exp(mi[idx1]  - mi_new)
        expmij = exp(mij - mi_new)
        li_idx1 = li[idx1]
        li_new = expmi*li_idx1 + expmij*lij

        # Oi_new = Pij ⊠ Vj
        # d may be larger than Bc, so we split the columns into 
        # blocks of Bc and use an offset.
        for o_col_offset=0i32:Bc:(d - 1i32)
            tj_new = o_col_offset + tj
            if tj_new <= d
                idx = CartesianIndex((ti, tj_new, tb))
                Oi_old  = Oi[idx]
                Oi_new = zero(T)
                for k=1:Bc
                    Oi_new += Pij[ti, k, tb]*Vj[k, tj_new, tb]
                end
                Oi[idx] = (li_idx1*expmi*Oi_old  + expmij*Oi_new) / li_new
            end
        end

        if tj == 1i32
            mi[ti, 1, tb] = mi_new
            li[ti, 1, tb] = li_new
        end
        sync_threads()
    end # end column tiles

    # write O, l, m
    @inbounds if tj == 1i32 && row <= N && bat <= B
        for k=1i32:d
            O[row, k, bat] = Oi[ti, k, tb]
        end
        l[row, 1, bat] = li[ti, 1, tb]
        m[row, 1, bat] = mi[ti, 1, tb]
    end
    return nothing
end

@inline function dense_fa(Q::CuArray{T, 3}, K::CuArray{T, 3}, V::CuArray{T, 3}) where T
    N, d, B = size(Q)
    return dense_fa!(similar(Q), similar(Q, N, 1, B), similar(Q, N, 1, B), Q, K, V)
end

function dense_fa!(
        O::CuArray{T, 3}, 
        l::CuArray{T, 3},
        m::CuArray{T, 3},
        Q::CuArray{T, 3}, 
        K::CuArray{T, 3}, 
        V::CuArray{T, 3}) where T
    N, d, B = size(Q)
    Br = 16
    Bc = Br
    Bb = 1

    threads = (Br, Bc, Bb)
    blocks = (cld(N, Br), 1, cld(B, Bb))
    shmem = sizeof(T)*Bb*(2*Br*(d + 1) + 2*Bc*d + Br*Bc)

    args = O, l, m, Q, K, V, Int32(N), Int32(d), Int32(B)
    @cuda blocks=blocks threads=threads shmem=shmem dense_fa_kernel!(args...)
    return O, l, m
end
