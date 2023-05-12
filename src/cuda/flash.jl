
function dense_fa_kernel!(
        O::CuDeviceMatrix{T}, 
        Q::CuDeviceMatrix{T}, 
        K::CuDeviceMatrix{T}, 
        V::CuDeviceMatrix{T}, 
        N::Int, d::Int, B::Int) where T
    ti, tj, tb = threadIdx().x, threadIdx().y, threadIdx().z
    Br, Bc, Bb = blockDim().x,  blockDim().y,  blockDim().z
    Tr, Tc, Tb = gridDim().x,   gridDim().y,   gridDim().z

    row = (blockIdx().x - 1)*Br + ti
    bat = (blockIdx().z - 1)*Bb + tb

    # load O, Q, m, l into shared memory
    Qi = CuDynamicSharedArray(T, (Br, d, Bb))
    Oi = CuDynamicSharedArray(T, (Br, d, Bb), sizeof(T)*Br*d*Bb)
    li = CuDynamicSharedArray(T, (Br, 1,  Bb), sizeof(T)*2*Br*d*Bb)
    mi = CuDynamicSharedArray(T, (Br, 1,  Bb), sizeof(T)*Bb*Br*(2*d + 1))

    if tj==1
        for k=1:d, b=1:Bb
            Qi[ti, k, bat] = row <= N && bat <= B ? Q[row, k] : zero(T)
            Oi[ti, k, bat] = zero(T)
            li[ti, 1, bat] = zero(T)
            mi[ti, 1, bat] = Inf(T)
        end
    end
    # make sure all data is loaded
    sync_threads()

    # allocate Kj, Vj
    Kj  = CuDynamicSharedArray(T, (Bc, d, Bb), sizeof(T)*2*Bb*Br*(d + 1))
    Vj  = CuDynamicSharedArray(T, (Bc, d, Bb), sizeof(T)*Bb*(2*Br*(d + 1) + Bc*d))
    Pij = CuDynamicSharedArray(T, (Br, Bc, Bb), sizeof(T)*Bb*(2*Br*(d + 1) + 2*Bc*d))

    for col_block_idx=1:Tc

    #   load K block, V block
    #   matmul blocks: P=QK^T
    #   compute m, l status
    #   matmul blocks: O=PV
    #   update l, m
    # store O, l, m
end


     
