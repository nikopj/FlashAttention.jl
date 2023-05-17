function softmax!(v::AnyCuVector{T}) where T
    n = length(v)

    function kernel(v, m, l, n)
        g  = this_grid()
        i0 = threadIdx().x + (blockIdx().x - 1i32)*blockDim().x

        # -- compute maximum --
        local_max = T(-Inf)

        # grid-stride loop
        i = i0
        while i <= n
            @inbounds local_max = max(local_max, v[i])
            i += blockDim().x * gridDim().x
        end

        block_max = CUDA.reduce_block(max, local_max, T(-Inf), Val(true))
        if threadIdx().x == 1i32
            @inbounds CUDA.@atomic m[] = max(m[], block_max)
        end
        sync_grid(g)
        m = m[]
        # -- done --

        # -- compute element-wise exponential and sum over vector ---
        # grid-stride loop
        local_sum = zero(T)
        i = i0
        while i <= n
            @inbounds v[i] = exp(v[i] - m)
            @inbounds local_sum += v[i]
            i += blockDim().x * gridDim().x
        end

        block_sum = CUDA.reduce_block(+, local_sum, zero(T), Val(true))
        if threadIdx().x == 1i32
            @inbounds CUDA.@atomic l[] += block_sum
        end
        sync_grid(g)
        l = l[]
        # -- done --

        # -- normalize vector -- 
        # grid-stride loop
        i = i0
        while i <= n
            @inbounds v[i] /= l
            i += blockDim().x * gridDim().x
        end
        # -- done --

        return nothing
    end

    m = similar(v, 1)
    l = CUDA.zeros(T, 1)
    fill!(m, T(-Inf))

    dev = device()

    wanted_threads = nextwarp(dev, n)
    function compute_threads(max_threads)
        if wanted_threads > max_threads
            prevwarp(dev, max_threads) 
        else
            wanted_threads
        end
    end

    # how many threads can we launch?
    args = v, m, l, n
    kernel  = @cuda launch=false kernel(args...)
    config  = launch_configuration(kernel.fun)
    threads = compute_threads(config.threads)
    blocks  = min(config.blocks, cld(n, config.blocks))
    kernel(args...; threads, blocks, cooperative=true)

    return v
end

function softmax!(V::AnyCuMatrix{T}) where T
    M, N = size(V)

    # Expectations:
    #   N is around 1024 to 4096.
    #   M is massive.
    #   A single block must therefore process multiple rows.

    # The program:
    #   Each block computes the softmax within a row.
    #   If the row is larger than a block, the block must iterate over columns.
    #   If the number of rows is greater than the grid dimension, each block must be 
    #   responsible for multiple rows.

    function kernel(V::CuDeviceMatrix{T}, M, N) where T
        # number of rows/cols to process per block
        nrow = cld(M, gridDim().x)  
        ncol = cld(N, blockDim().x)

        m = CuStaticSharedArray(T, 1)
        l = CuStaticSharedArray(T, 1)
        buf = CuDynamicSharedArray(T, ncol*blockDim().x)

        # grid-stride loop
        for row = blockIdx().x : gridDim().x : M
            # -----------------------------------------------
            # -- load data into buffer and compute maximum --
            # -----------------------------------------------
            local_max = T(-Inf)

            # block-stride loop
            for col = threadIdx().x : blockDim().x : ncol*blockDim().x
                @inbounds buf[col] = col <= N ? V[row, col] : T(-Inf)
                @inbounds local_max = max(local_max, buf[col])
            end
            row_max = CUDA.reduce_block(max, local_max, T(-Inf), Val(true))

            # reduce_block gathers to thread-1,
            # so we use shared memory to scatter it.
            if threadIdx().x == 1i32
                m[] = row_max
            end
            sync_threads()
            row_max = m[]
            # ----------
            # -- done --
            # ----------

            # -----------------
            # -- compute sum --
            # -----------------
            local_sum = zero(T)

            # block-stride loop
            for col = threadIdx().x : blockDim().x : ncol*blockDim().x
                @inbounds e = exp(buf[col] - row_max)
                @inbounds buf[col] = e
                local_sum += e
            end
            row_sum = CUDA.reduce_block(+, local_sum, zero(T), Val(true))

            if threadIdx().x == 1i32
                l[] = row_sum
            end
            sync_threads()
            row_sum = l[]
            # ----------
            # -- done --
            # ----------
             
            # ----------------------
            # -- normalize vector -- 
            # ----------------------
            # block-stride loop
            for col = threadIdx().x : blockDim().x : ncol*blockDim().x
                if col <= N
                    @inbounds V[row, col] = buf[col] / row_sum
                end
            end
            # ----------
            # -- done --
            # ----------
        end
        return nothing
    end

    dev = device()
    wanted_threads = nextwarp(dev, N)
    function compute_threads(max_threads)
        if wanted_threads > max_threads
            prevwarp(dev, max_threads) 
        else
            wanted_threads
        end
    end
    compute_shmem(threads) = sizeof(T)*cld(N, threads)*threads

    # how many threads can we launch?
    args = V, Int32(M), Int32(N)
    kernel  = @cuda launch=false kernel(args...)
    config  = launch_configuration(kernel.fun; shmem=compute_shmem∘compute_threads)
    threads = compute_threads(config.threads)
    blocks  = min(config.blocks, M)
    shmem   = compute_shmem(threads)
    kernel(args...; threads=threads, blocks=blocks, shmem=shmem)
    return V
end
