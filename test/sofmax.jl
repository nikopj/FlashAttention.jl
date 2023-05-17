using CUDA
using CUDA: i32

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

    function kernel(V, M, N)
        row0 = blockIdx().x
        col0 = threadIdx().x 

        # loop over rows for which this block is responsible for
        nrow = cld(M, gridDim().x)

        m = CuStaticSharedArray(T, 1)
        l = CuStaticSharedArray(T, 1)

        for drow=0:nrow-1
            row = row0 + drow * gridDim().x
            if row > M
                break
            end

            # ---------------------
            # -- compute maximum --
            # ---------------------
            local_max = T(-Inf)

            # block-stride loop
            col = col0
            while col <= N
                @inbounds local_max = max(local_max, V[row, col])
                col += blockDim().x 
            end
            row_max = CUDA.reduce_block(max, local_max, T(-Inf), Val(true))

            # reduce_block gathers to thread-1, so we use shared memory to scatter it.
            if threadIdx().x == 1i32
                m[] = row_max
            end
            sync_threads()
            # ----------
            # -- done --
            # ----------

            # -----------------
            # -- compute sum --
            # -----------------
            local_sum = zero(T)

            # block-stride loop
            col = col0
            while col <= N
                @inbounds e = exp(V[row, col] - m[])
                @inbounds V[row, col] = e
                local_sum += e
                col += blockDim().x 
            end
            row_sum = CUDA.reduce_block(+, local_sum, zero(T), Val(true))

            if threadIdx().x == 1i32
                l[] = row_sum
            end
            sync_threads()
            # ----------
            # -- done --
            # ----------
             
            # ----------------------
            # -- normalize vector -- 
            # ----------------------
            # block-stride loop
            col = col0
            while col <= N
                @inbounds V[row, col] /= l[]
                col += blockDim().x 
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

    # how many threads can we launch?
    args = V, Int32(M), Int32(N)
    kernel  = @cuda launch=false kernel(args...)
    config  = launch_configuration(kernel.fun)
    threads = compute_threads(config.threads)
    blocks  = min(config.blocks, M)
    kernel(args...; threads, blocks)
    return V
end
