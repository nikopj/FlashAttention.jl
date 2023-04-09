
function dense_fa(q::AbstractArray{T, D}, k::AbstractArray{T, D}, v::AbstractArray{T, D}) where {T, D}
    N, batchsize = size(q, 1), size(q, D)
    @assert N == size(k, 1) && N == size(v, 1)

    d  = size(q, D-1)
    dv = size(v, D-1)

    M  = 32000 # SRAM size

    # row/column block-length
    Br = min(d, cld(M, 4*d*batchsize))
    Bc = cld(M, 4*d*batchsize)

    # num row/column blocks
    Tr = cld(N, Br)
    Tc = cld(N, Bc)

    #@show Br, Bc, Tr, Tc

    # initialize variables
    Q = reshape(q, :, d, batchsize)
    K = reshape(k, :, d, batchsize)
    V = reshape(v, :, dv,  batchsize)
    O = similar(V)
    l = similar(q, N, 1, batchsize)
    m = similar(q, N, 1, batchsize)

    fill!(O, zero(T))
    fill!(l, zero(T))
    fill!(m, T(-Inf))

    @threads for j=1:Tc
        start_jdx = (j-1)*Bc + 1
        end_jdx = min(N, j*Bc)
        Kj = K[start_jdx:end_jdx, :, :]
        Vj = V[start_jdx:end_jdx, :, :]

        for i=1:Tr
            start_idx = (i-1)*Br + 1
            end_idx = min(N, i*Br)
            Qi = Q[start_idx:end_idx, :, :]
            li = l[start_idx:end_idx, :, :]
            mi = m[start_idx:end_idx, :, :]
            Oi = O[start_idx:end_idx, :, :]

            Sij = (Qi ⊠ batched_transpose(Kj)) ./ T(sqrt(d))  # similarity matrix
            mij = maximum(Sij, dims=2)                          

            Pij = @. exp(Sij - mij)                             # adjacency matrix
            lij = sum(Pij, dims=2)

            mi_new = max.(mi, mij)
            li_new = @. exp(mi - mi_new)*li + exp(mij - mi_new)*lij

            @. Oi = (li*exp(mi - mi_new)*Oi + exp(mij - mi_new)*$batched_mul(Pij, Vj)) / li_new

            # write back to memory
            l[start_idx:end_idx, :, :] = li_new
            m[start_idx:end_idx, :, :] = mi_new
            O[start_idx:end_idx, :, :] = Oi
        end
    end

    y = reshape(O, size(q)[1:D-2]..., dv, :)
    return y, m, l
end

