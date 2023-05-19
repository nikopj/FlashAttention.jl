function fused_softmax!(v::AnyCuVector{T}) where T
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

@inline function fused_softmax!(V::AnyCuMatrix{T}; dims=1) where T
    @assert dims in (1, 2) "dims=$dims must be 1 or 2"
    return dims == 1 ? fused_col_softmax!(V) : fused_row_softmax!(V)
end

function fused_row_softmax!(V::AnyCuMatrix{T}) where T
    M, N = size(V)

    # Expectations:
    #   N is around 1024 to 4096, and N * sizeof(T) fits in Shared Memory
    #   M is massive.

    # The program:
    #   Each block computes the softmax within a row.
    #   If the row is larger than a block, the block must iterate over columns.
    #   If the number of rows is greater than the grid dimension, each block must be 
    #   responsible for multiple rows.

    function kernel!(V, M, N, row0) 
        # number of rows/cols to process per block
        nrow = cld(M, gridDim().x)  
        ncol = cld(N, blockDim().x)

        m = CuStaticSharedArray(T, 1)
        l = CuStaticSharedArray(T, 1)
        buf = CuDynamicSharedArray(T, ncol*blockDim().x)

        # grid-stride loop
        @inbounds for row = (row0 + blockIdx().x - 1) : gridDim().x : (row0 + M - 1)
            # -----------------------------------------------
            # -- load data into buffer and compute maximum --
            # -----------------------------------------------
            local_max = T(-Inf)

            # block-stride loop
            @unroll for col = threadIdx().x : blockDim().x : ncol*blockDim().x
                buf[col] = col <= N ? V[row, col] : T(-Inf)
                local_max = max(local_max, buf[col])
            end
            row_max = CUDA.reduce_block(max, local_max, T(-Inf), Val(true))

            # reduce_block gathers to thread-1,
            # so we use shared memory to scatter it.
            if threadIdx().x == 1i32
                m[] = row_max
            end
            sync_threads()
            row_max = m[]

            # -----------------
            # -- compute sum --
            # -----------------
            local_sum = zero(T)

            # block-stride loop
            @unroll for col = threadIdx().x : blockDim().x : ncol*blockDim().x
                e = exp(buf[col] - row_max)
                buf[col] = e
                local_sum += e
            end
            row_sum = CUDA.reduce_block(+, local_sum, zero(T), Val(true))

            if threadIdx().x == 1i32
                l[] = row_sum
            end
            sync_threads()
            row_sum = l[]
             
            # ----------------------
            # -- normalize vector -- 
            # ----------------------
            # block-stride loop
            @unroll for col = threadIdx().x : blockDim().x : ncol*blockDim().x
                if col <= N
                    V[row, col] = buf[col] / row_sum
                end
            end
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

    # when we have a large number of rows, 
    # parallelize over streams as well.
    max_rows = 2^16
    num_streams = cld(M, max_rows)
    streams = [CuStream(; flags=CUDA.STREAM_NON_BLOCKING) for _ in 1:num_streams]
    kernel  = @cuda launch=false kernel!(V, Int32(M), Int32(N), 1i32)

    @unroll for s=1:num_streams
        row_range = ((s-1)*max_rows + 1):min(M, s*max_rows)
        Ms = length(row_range)

        config  = launch_configuration(kernel.fun; shmem=compute_shmem∘compute_threads)
        threads = compute_threads(config.threads)
        # blocks  = min(config.blocks, Ms)
        blocks  = Ms
        shmem   = compute_shmem(threads)
        kernel(V, Int32(Ms), Int32(N), Int32(row_range[1]); 
               threads=threads, blocks=blocks, shmem=shmem, stream=streams[s])
    end
    for s in streams; CUDA.synchronize(s); end

    return V
end

function fused_col_softmax!(V::AnyCuMatrix{T}) where T
    M, N = size(V)

    function kernel!(V, M, N, col0) 
        # number of rows/cols to process per block
        ncol = cld(N, gridDim().x)  
        nrow = cld(M, blockDim().x)

        m = CuStaticSharedArray(T, 1)
        l = CuStaticSharedArray(T, 1)
        buf = CuDynamicSharedArray(T, nrow*blockDim().x)

        # grid-stride loop
        @inbounds for col = (col0 + blockIdx().x - 1) : gridDim().x : (col0 + N - 1)
            # -----------------------------------------------
            # -- load data into buffer and compute maximum --
            # -----------------------------------------------
            local_max = T(-Inf)

            # block-stride loop
            @unroll for row = threadIdx().x : blockDim().x : nrow*blockDim().x
                buf[row] = row <= M ? V[row, col] : T(-Inf)
                local_max = max(local_max, buf[row])
            end
            col_max = CUDA.reduce_block(max, local_max, T(-Inf), Val(true))

            # reduce_block gathers to thread-1,
            # so we use shared memory to scatter it.
            if threadIdx().x == 1i32
                m[] = col_max
            end
            sync_threads()
            col_max = m[]

            # -----------------
            # -- compute sum --
            # -----------------
            local_sum = zero(T)

            # block-stride loop
            @unroll for row = threadIdx().x : blockDim().x : nrow*blockDim().x
                e = exp(buf[row] - col_max)
                buf[row] = e
                local_sum += e
            end
            col_sum = CUDA.reduce_block(+, local_sum, zero(T), Val(true))

            if threadIdx().x == 1i32
                l[] = col_sum
            end
            sync_threads()
            col_sum = l[]
             
            # ----------------------
            # -- normalize vector -- 
            # ----------------------
            # block-stride loop
            @unroll for row = threadIdx().x : blockDim().x : nrow*blockDim().x
                if row <= M
                    V[row, col] = buf[row] / col_sum
                end
            end
        end
        return nothing
    end

    dev = device()
    wanted_threads = nextwarp(dev, M)
    function compute_threads(max_threads)
        if wanted_threads > max_threads
            prevwarp(dev, max_threads) 
        else
            wanted_threads
        end
    end
    compute_shmem(threads) = sizeof(T)*cld(M, threads)*threads

    # when we have a large number of rows, 
    # parallelize over streams as well.
    max_cols = 2^16
    num_streams = cld(N, max_cols)
    streams = [CuStream(; flags=CUDA.STREAM_NON_BLOCKING) for _ in 1:num_streams]
    kernel  = @cuda launch=false kernel!(V, Int32(M), Int32(N), 1i32)

    @unroll for s=1:num_streams
        col_range = ((s-1)*max_cols + 1):min(N, s*max_cols)
        Ns = length(col_range)

        config  = launch_configuration(kernel.fun; shmem=compute_shmem∘compute_threads)
        threads = compute_threads(config.threads)
        # blocks  = min(config.blocks, Ns)
        blocks  = Ns
        shmem   = compute_shmem(threads)
        kernel(V, Int32(M), Int32(Ns), Int32(col_range[1]); 
               threads=threads, blocks=blocks, shmem=shmem, stream=streams[s])
    end
    for s in streams; CUDA.synchronize(s); end

    return V
end