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

        nrow = cld(M, gridDim().x)
        row_end = min(M, row0 + nrow -1)

        # m = CuDynamicSharedArray(T, nrow)
        # l = CuDynamicSharedArray(T, nrow, sizeof(m))

        # ---------------------
        # -- compute maximum --
        # ---------------------
        # we compute the maximum over both rows (for which this block is responsible) 
        # and columns, as it is only used for numerical stability.
        local_max = T(-Inf)

        # block-stride loop
        j = col0
        while j <= N
            for i = row0:row_end
                @inbounds local_max = max(local_max, V[i, j])
            end
            j += blockDim().x 
        end
        block_max = CUDA.reduce_block(max, local_max, T(-Inf), Val(true))
        # ----------
        # -- done --
        # ----------

        # -----------------
        # -- compute sum --
        # -----------------
        # we additionally compute the element-wise exponential
         
        local_sum = CUDA.zeros(T, nrow)
        block_sum = CUDA.zeros(T, nrow)

        # block-stride loop
        j = col0
        while j <= N
            for i = row0:row_end
                @inbounds V[i, j] = exp(V[i, j] - block_max)
                @inbounds local_sum[i] += V[i, j]
            end
            j += blockDim().x 
        end

        for i = row0:row_end
            block_sum[i] = CUDA.reduce_block(+, local_sum[i], zero(T), Val(true))
        end
        # ----------
        # -- done --
        # ----------

        # ----------------------
        # -- normalize vector -- 
        # ----------------------
         
        # block-stride loop
        j = col0
        while j <= N
            for i = row0:row_end
                @inbounds V[i, j] /= block_sum[i]
            end
            j += blockDim().x 
        end
        # ----------
        # -- done --
        # ----------
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
    args = v, M, N
    kernel  = @cuda launch=false kernel(args...)
    config  = launch_configuration(kernel.fun)
    threads = compute_threads(config.threads)
    blocks  = min(config.blocks, cld(M, config.blocks))
    kernel(args...; threads, blocks, cooperative=true)

    return v
end
