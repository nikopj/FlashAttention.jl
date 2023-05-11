
function dense_fa_kernel!(
        O::CuDeviceMatrix{T}, 
        Q::CuDeviceMatrix{T}, 
        K::CuDeviceMatrix{T}, 
        V::CuDeviceMatrix{T}, 
        N::Int, d::Int, B::Int) where T

    Br, Bc, Bb = blockDim().x, blockDim().y, blockDim().z
    ti, tj, tb = threadIdx().x, threadIdx().y, threadIdx().z

    # load O, Q, m, l into shared memory

    # for k in NUM_BLOCKS
    #   load K block, V block
    #   matmul blocks: P=QK^T
    #   compute m, l status
    #   matmul blocks: O=PV
    #   update l, m
    # store O, l, m
end


     
