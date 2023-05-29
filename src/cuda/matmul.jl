using CUDA: i32

function matvec!(y::AnyCuVector{T}, A::AnyCuMatrix{T}, x::AnyCuVector{T}) where T
    M, N = size(A)

    # program:
    #   - a single WARP is for a row of A <--> a scalar output of y
    function kernel_warp_impl!(y, A, x, M, N)
        tid = threadIdx().x + (blockIdx().x - 1i32)*blockDim().x # thread id
        wid = cld(tid, warpsize())  # warp-id
        lid = mod1(tid, warpsize()) # lane-id

        # grid-stride loop
        @inbounds for row = wid : cld(blockDim().x, warpsize()) * gridDim().x : M
            # warp-stride loop
            acc = zero(T)
            for k = lid : warpsize() : N
                acc += A[row, k] * x[k]
            end
            acc = CUDA.reduce_warp(+, acc)

            if row <= M && lid == 1
                y[row] = acc
            end
        end
        return nothing
    end

    # program:
    #   - a single BLOCK is for a row of A <--> a scalar output of y
    function kernel_block_impl!(y, A, x, M, N)
        # grid-stride loop
        @inbounds for row = blockIdx().x : gridDim().x : M
            acc = zero(T)
            for k = threadIdx().x : blockDim().x : N
                acc += A[row, k] * x[k]
            end
            acc = CUDA.reduce_block(+, acc, zero(T), Val(true))

            if threadIdx().x == 1 && row <= M
                y[row] = acc
            end
        end
        return nothing
    end

    # program:
    #   - a single WARP is for a row of A <--> a scalar output of y
    #   - a single block uses shared memory to cache X
    function kernel_shwarp_impl!(y, A, x, M, N)
        tid = threadIdx().x + (blockIdx().x - 1i32)*blockDim().x # thread id
        wid = cld(tid, warpsize())  # warp-id
        lid = mod1(tid, warpsize()) # lane-id

        x_cache = CuDynamicSharedArray(T, blockDim().x)

        # grid-stride loop (over rows of A)
        @inbounds for row = wid : cld(blockDim().x, warpsize()) * gridDim().x : M
            acc = zero(T)
            # tile loop (over cols of A)
            for n = 1i32 : cld(N, blockDim().x)
                # note: the element of x loaded into cache (just 1) by this thread 
                # are different than those columns used in the warp-loop
                load_col = threadIdx().x + (n - 1i32)*blockDim().x

                x_cache[threadIdx().x] = load_col <= N ? x[load_col] : zero(T)
                sync_threads() # ensure cache data present

                # warp loop
                col0 = (n - 1i32)*blockDim().x
                for k = lid : warpsize() : blockDim().x
                    q = k + col0
                    Aval = q <= N ? A[row, q] : zero(T)
                    acc += Aval * x_cache[k]
                end
                sync_threads() # ensure cache data used before overwriting on next loop
            end
            yval = CUDA.reduce_warp(+, acc)

            if row <= M && lid == 1
                y[row] = yval
            end
        end
        return nothing
    end

    kernelfun = kernel_warp_impl!

    dev = device()

    wanted_threads = nextwarp(dev, N)
    function compute_threads(max_threads)
        if wanted_threads > max_threads
            prevwarp(dev, max_threads) 
        else
            wanted_threads
        end
    end
    compute_shmem(threads) = (kernelfun == kernel_shwarp_impl!) ? sizeof(T)*threads : 0

    args = y, A, x, Int32(M), Int32(N)
    kernel = @cuda launch=false kernelfun(args...)
    config  = launch_configuration(kernel.fun; shmem=compute_shmem∘compute_threads)
    threads = compute_threads(config.threads)
    blocks = if kernelfun in (kernel_warp_impl!, kernel_shwarp_impl!) # == kernel_warp_impl! || kernelfun == kernel_shwap_impl!
        cld(M, cld(threads, warpsize(dev))) # rows / rows_per_block
    elseif kernelfun == kernel_block_impl!
        M
    end
    shmem = compute_shmem(threads)
    kernel(args...; threads=threads, blocks=blocks, shmem=shmem)
    return y
end

