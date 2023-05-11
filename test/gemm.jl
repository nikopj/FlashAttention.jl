using CUDA, LinearAlgebra

function matmul_naive_kernel!(
        C::CuDeviceMatrix{T}, 
        A::CuDeviceMatrix{T}, 
        B::CuDeviceMatrix{T}, 
        M::Int, N::Int, K::Int) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y 

    @inbounds if i <= M && j <= N
        sum = zero(T)
        for k in 1:K
            sum += A[i, k] * B[k, j]
        end
        C[i, j] = sum
    end
    return nothing
end

function matmul_naive!(C::CuMatrix{T}, A::CuMatrix{T}, B::CuMatrix{T}) where T
    M, N = size(C)
    K = size(A, 2)

    function get_threads(threads)
        threads_x = floor(Int, sqrt(threads))
        threads_y = threads ÷ threads_x
        return (threads_x, threads_y)
    end

    args = C, A, B, M, N, K
    kernel = @cuda launch=false matmul_naive_kernel!(args...)
    config = launch_configuration(kernel.fun) #max_threads=256)

    threads = get_threads(config.threads)
    blocks = cld.((M, N), threads)
    kernel(args...; threads=threads, blocks=blocks)
    return C
end

function matmul_tiled_kernel!(
        C::CuDeviceMatrix{T}, 
        A::CuDeviceMatrix{T}, 
        B::CuDeviceMatrix{T}, 
        M::Int, N::Int, K::Int) where T
    ti = threadIdx().x 
    tj = threadIdx().y 

    # the tile is size TILE_SIZE x TILE_SIZE
    TILE_SIZE = blockDim().x

    # row and col of output matrix, 
    # hence TILE_SIZE for both
    row = (blockIdx().x - 1)*TILE_SIZE + ti
    col = (blockIdx().y - 1)*TILE_SIZE + tj

    a = CuDynamicSharedArray(T, (TILE_SIZE, TILE_SIZE))
    # 3rd argument is the offset b needs to be from a
    b = CuDynamicSharedArray(T, (TILE_SIZE, TILE_SIZE), sizeof(T)*TILE_SIZE*TILE_SIZE)

    cval = zero(T)
    for k = 1:cld(K, TILE_SIZE)
        # load A tile
        if row <= M && (k-1)*TILE_SIZE + tj <= K
            @inbounds a[ti, tj] = A[row, (k-1)*TILE_SIZE + tj]
        else
            @inbounds a[ti, tj] = zero(T)
        end

        # load B tile
        if (k-1)*TILE_SIZE + ti <= K && col <= N
            @inbounds b[ti, tj] = B[(k-1)*TILE_SIZE + ti, col]
        else
            @inbounds b[ti, tj] = zero(T)
        end

        # sync threads within the thread-block block
        # to ensure a, b are all loaded
        sync_threads()

        # matmul tiles
        for k = 1:TILE_SIZE
            @inbounds cval += a[ti, k] * b[k, tj]
        end

        # sync to ensure all values are used before 
        # being overwritten on next loop iteration
        sync_threads()
    end

    if row <= M && col <= N
        @inbounds C[row, col] = cval
    end
    return nothing
end

function matmul_tiled!(C::CuMatrix{T}, A::CuMatrix{T}, B::CuMatrix{T}) where T
    M, N = size(C)
    K = size(A, 2)
    TILE_SIZE = 16

    threads = (TILE_SIZE, TILE_SIZE)
    blocks = cld.((M, N), TILE_SIZE)
    shmem = 2 * prod(threads) * sizeof(T)

    args = C, A, B, M, N, K
    @cuda blocks=blocks threads=threads shmem=shmem matmul_tiled_kernel!(args...)
    return C
end
