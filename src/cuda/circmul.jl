function circmul!(C::AnyCuMatrix{T}, A::AnyCuMatrix{T}, B::AnyCuMatrix{T}) where T
    N, d = size(B)
    W, _ = size(A)

    function kernel!(C, A, B, N, W, d)
        # kernel:
        #   - each thread is responsible for the i,j entry of the output matrix
    end
end

function circmul!(C::AnyCuMatrix{T}, A::AnyCuMatrix{T}, B::AnyCuMatrix{T}) where T
    N, d = size(B)
    W, _ = size(A)

    function kernel!(C, A, B, N, W, d)
        # expectations:
        #   - A is a W x N matrix which stores with non-zero elements of a 
        #     N x N circulant matrix, each column being a row of the circulant A (circA).
        #   - B is a N x d matrix
        #   - N is very large 
        #   - d is <= 4096
        #   - W is <= 4096
        
        # kernel:
        #   - each block with be responsible for a TILE_SIZE x TILE_SIZE block of output C
        
        TILE_SIZE = min(blockDim().x, blockDim().y)
        tix = threadIdx().x
        tiy = threadIdx().y

        row0 = threadIdx().x + TILE_SIZE * (blockIdx().x - 1i32)
        col0 = threadIdx().y + TILE_SIZE * (blockIdx().y - 1i32)

        tileA = CuDynamicSharedArray(T, (TILE_SIZE, TILE_SIZE))
        tileB = CuDynamicSharedArray(T, (TILE_SIZE, TILE_SIZE), sizeof(tileA))

        # 2D grid-stride loop
        @inbounds for row = row0 : TILE_SIZE*gridDim().x : N, col = col0 : TILE_SIZE*gridDim().y : d
            cval = zero(T)

            # loop over num-tiles in A
            for n = 1i32 : cld(W, TILE_SIZE)
                # -----------------------
                # -- load tileA, tileB --
                # -----------------------
                j = threadIdx().x + (n-1i32)*TILE_SIZE
                i = threadIdx().y + (n-1i32)*TILE_SIZE 
                
                tileA[tix, tiy] = (row <= N && j <= W) ? A[j, row] : zero(T)
                tileB[tix, tiy] = (i <= W && col <= d) ? B[cartesian_circulant(i, W, N), col] : zero(T)
                sync_threads()

                @unroll for k = 1i32 : TILE_SIZE
                    nothing
                end
                    
            end
        end
    end

end
