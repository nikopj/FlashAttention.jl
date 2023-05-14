using CUDA, LinearAlgebra
using KernelAbstractions.Extras: @unroll
using CUDA: i32

function matmul_naive_kernel!(
        C::CuDeviceMatrix{T}, 
        A::CuDeviceMatrix{T}, 
        B::CuDeviceMatrix{T}, 
        M::Int32, N::Int32, K::Int32) where T
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x 
    j = (blockIdx().y - 1i32) * blockDim().y + threadIdx().y 

    @inbounds if i <= M && j <= N
        sum = zero(T)
        for k in 1i32:K
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

    args = C, A, B, Int32(M), Int32(N), Int32(K)
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
        M::Int32, N::Int32, K::Int32) where T
    ti = threadIdx().x 
    tj = threadIdx().y 

    # the tile is size TILE_SIZE x TILE_SIZE
    TILE_SIZE = blockDim().x

    # row and col of output matrix
    row = (blockIdx().x - 1i32)*TILE_SIZE + ti
    col = (blockIdx().y - 1i32)*TILE_SIZE + tj

    a = CuDynamicSharedArray(T, (TILE_SIZE, TILE_SIZE))
    b = CuDynamicSharedArray(T, (TILE_SIZE, TILE_SIZE), sizeof(T)*TILE_SIZE*TILE_SIZE)

    cval = zero(T)
    for k = 1i32:cld(K, TILE_SIZE)
        # load A tile
        a_col = (k-1)*TILE_SIZE + tj
        if row <= M &&  a_col <= K
            @inbounds a[ti, tj] = A[row, a_col]
        else
            @inbounds a[ti, tj] = zero(T)
        end

        # load B tile
        b_row = (k-1)*TILE_SIZE + ti
        if b_row <= K && col <= N
            @inbounds b[ti, tj] = B[b_row, col]
        else
            @inbounds b[ti, tj] = zero(T)
        end

        # sync threads within the thread-block block
        # to ensure a, b are all loaded
        sync_threads()

        # matmul tiles
        @unroll for k = 1i32:TILE_SIZE
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

    args = C, A, B, Int32(M), Int32(N), Int32(K)
    @cuda blocks=blocks threads=threads shmem=shmem matmul_tiled_kernel!(args...)
    return C
end

function matmul_tiled_kernel2!(
        C::CuDeviceMatrix{T}, 
        A::CuDeviceMatrix{T}, 
        B::CuDeviceMatrix{T}, 
        M::Int32, N::Int32, K::Int32,
        TILE_WIDTH::Int32, R::Int32) where T
    ti = threadIdx().x 
    tj = threadIdx().y 

    # note that each thread is now responsible for a RxR submatrix.
    # assume blockDim().y == blockDim().x
    # and that TILE_WIDTH is a multiple of blockDim
    R = TILE_WIDTH ÷ blockDim().x

    # starting row and col of output matrix
    row0 = (blockIdx().x - 1i32)*TILE_WIDTH + (ti - 1i32)*R + 1i32
    col0 = (blockIdx().y - 1i32)*TILE_WIDTH + (tj - 1i32)*R + 1i32

    a = CuDynamicSharedArray(T, (TILE_WIDTH, TILE_WIDTH))
    b = CuDynamicSharedArray(T, (TILE_WIDTH, TILE_WIDTH), sizeof(T)*TILE_WIDTH*TILE_WIDTH)
    c = CuDynamicSharedArray(T, (TILE_WIDTH, TILE_WIDTH), sizeof(T)*2*TILE_WIDTH*TILE_WIDTH)

    # get tile indices for ii=1i32:R, jj=1i32:R submatrix of current thread at block coord (ti, tj)
    subidx(ii, jj) = CartesianIndex(((ti-1i32)*R + ii, (tj-1i32)*R + jj))

    @unroll for ii=1i32:R
        @unroll for jj=1i32:R
            @inbounds c[subidx(ii, jj)] = zero(T)
        end
    end

    # loop over tiles (several tiles in a row block of A, a col block of B)
    for t = 1i32:cld(K, TILE_WIDTH)
        # load A tile
        @unroll for ii=1i32:R
            @unroll for jj=1i32:R
                row = row0 + ii - 1i32
                col = (t-1i32)*TILE_WIDTH + (tj-1i32)*R + jj
                if row <= M && col <= K
                    @inbounds a[subidx(ii,jj)] = A[row, col]
                else
                    @inbounds a[subidx(ii,jj)] = zero(T)
                end
            end
        end

        # load B tile
        @unroll for ii=1i32:R
            @unroll for jj=1i32:R
                row = (t-1i32)*TILE_WIDTH + (ti-1i32)*R + ii
                col = col0 + jj - 1i32
                if row <= K && col <= N
                    @inbounds b[subidx(ii,jj)] = B[row, col]
                else
                    @inbounds b[subidx(ii,jj)] = zero(T)
                end
            end
        end
        # sync threads within the thread-block block
        # to ensure a, b are all loaded
        sync_threads()

        # compute inner products for each element in RxR submat
        for k = 1i32:TILE_WIDTH
            @unroll for ii=1i32:R
                @unroll for jj=1i32:R
                    m, n = subidx(ii, jj).I
                    @inbounds c[m, n] += a[m, k] * b[k, n]
                end
            end
        end

        # sync to ensure all values are used before 
        # being overwritten on next loop iteration
        sync_threads()
    end

    @unroll for ii=1i32:R
        @unroll for jj=1i32:R
            row = row0 + ii - 1i32
            col = col0 + jj - 1i32
            if row <= M && col <= N
                @inbounds C[row, col] = c[subidx(ii, jj)]
            end
        end
    end
    return nothing
end

function matmul_tiled2!(C::CuMatrix{T}, A::CuMatrix{T}, B::CuMatrix{T}) where T
    M, N = size(C)
    P = min(M, N)
    K = size(A, 2)

    blockDim = 16
    threads = (blockDim, blockDim)
    TILE_WIDTH = 2*blockDim
    blocks = cld.((M, N), TILE_WIDTH)
    shmem = 3 * TILE_WIDTH^2 * sizeof(T)

    args = C, A, B, Int32(M), Int32(N), Int32(K), Int32(TILE_WIDTH)
    @cuda blocks=blocks threads=threads shmem=shmem matmul_tiled_kernel2!(args...)
    return C
end
