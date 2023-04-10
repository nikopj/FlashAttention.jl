# m: filter index, 
# s: shift
# M: filter-length
circshift_index(m, s, M) = mod(m - 1 - s, M) + 1

function cartesian_circulant(n, N, M)
    # filter size must be odd
    p = (M-1) ÷ 2
    j = cld(n, M) # col num
    m = mod(n-1, M) + 1
    if j <= p
        m = circshift_index(m, j - p - 1, M)
    elseif j > N-p 
        m = circshift_index(m, p - N + j, M)
    end
    i = mod((m-1) + (j-1) - p, N) + 1
    return i, j
end

function cartesian_circulant(n, N1, N2, M)
    # filter size must be odd
    p = (M-1) ÷ 2
    j = cld(n, M^2)                            # global col num
    jj = cld(j, N1)                            # block col num
    j0 = mod(j-1, N1) + 1                      # intra block col num
    nn = mod(cld(n, M) - 1, M) + 1 + M*(jj-1)  # block num
    mm = mod(nn-1, M) + 1                      # block filter coeff num
    m0 = n - M*(mm-1) - (j-1)*M^2              # intra block col filter coeff num
    if jj <= p
        mm = circshift_index(mm, jj - p - 1, M)
    elseif jj > N2-p
        mm = circshift_index(mm, p - N2 + jj, M)
    end
    if j0 <= p
        m0 = circshift_index(m0, j0 - p - 1, M)
    elseif j0 > N1-p
        m0 = circshift_index(m0, p - N1 + j0, M)
    end
    ii = mod((mm-1) + (jj-1) - p, N2) + 1              # block row num
    i  = N1*(ii-1) + mod((m0-1) + (j0-1) - p, N1) + 1  # rownum
    return i, j
end

function circulant(N::Int, M::Int, Tv=Float64, Ti=Int64) 
    rowval = (Ti∘first∘cartesian_circulant).(1:N*M, N, M) 
    colptr = 1 .+ M .* collect(0:N) .|> Ti
    return SparseMatrixCSC{Tv, Ti}(N, N, colptr, rowval, ones(Tv, N*M))
end

function circulant((N1, N2)::Tuple{Int, Int}, M::Int, Tv=Float64, Ti=Int64) 
    rowval = (Ti∘first∘cartesian_circulant).(1:N1*N2*M*M, N1, N2, M)
    colptr = 1 .+ M^2 .* collect(0:N1*N2) .|> Ti
    return SparseMatrixCSC{Tv, Ti}(N1*N2, N1*N2, colptr, rowval, ones(Tv, N1*N2*M*M))
end

function circulant_kron((N1, N2)::Tuple{Int, Int}, M::Int, Tv=Float64, Ti=Int64) 
    A = circulant(N2, M, Tv, Ti)
    B = circulant(N1, M, Tv, Ti)
    return kron(A, B)
end

function cucirculant_kron((N1, N2)::Tuple{Int, Int}, M::Int, Tv=Float32, Ti=Int32) 
    A = cucirculant(N2, M, Tv, Ti)
    B = cucirculant(N1, M, Tv, Ti)
    return kron(A, B)
end

function cucirculant_kernel!(colval::AbstractArray{T}, N, M, maxidx) where T
    n = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds if n <= maxidx
        i = cartesian_circulant(n, N, M)[1]
        colval[n] = T(i)
    end
    return nothing
end

function cucirculant_kernel!(colval::AbstractArray{T}, N1, N2, M, maxidx) where T
    n = (blockIdx().x-1) * blockDim().x + threadIdx().x
    @inbounds if n <= maxidx
        i = cartesian_circulant(n, N1, N2, M)[1]
        colval[n] = T(i)
    end
    return nothing
end

function cucirculant(N::Int, M::Int, Tv=Float32, Ti=Int32) 
    @assert M % 2 == 1 "filter size M=$M must be odd."

    colval = CuVector{Ti}(undef, N*M)
    nzval  = CUDA.ones(Tv, N*M)
    rowptr = CuVector{Ti}(undef, N + 1)

    @. rowptr = 0:N
    @. rowptr *= M
    @. rowptr += 1

    maxidx = N*M
    args = colval, N, M, maxidx
    kernel = @cuda launch=false cucirculant_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(maxidx, config.threads)
    blocks = cld(maxidx, threads)
    kernel(args...; threads=threads, blocks=blocks)

    return CuSparseMatrixCSR{Tv, Ti}(rowptr, colval, nzval, (N, N))
end

function cucirculant((N1, N2)::Tuple{Int, Int}, M::Int, Tv=Float32, Ti=Int32) 
    @assert M % 2 == 1 "filter size M=$M must be odd."

    colval = CuVector{Ti}(undef, N1*N2*M^2)
    nzval  = CUDA.ones(Tv, N1*N2*M^2)
    rowptr = CuVector{Ti}(undef, N1*N2 + 1)

    @. rowptr = 0:(N1*N2)
    @. rowptr *= M^2
    @. rowptr += 1

    maxidx = N1*N2*M^2
    args = colval, N1, N2, M, maxidx
    kernel = @cuda launch=false cucirculant_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(maxidx, config.threads)
    blocks = cld(maxidx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return CuSparseMatrixCSR{Tv, Ti}(rowptr, colval, nzval, (N1*N2, N1*N2))
end
    
circulant(x::AnyCuArray{T, 4}, M::Int) where {T} = cucirculant(size(x)[1:2], M, T)
circulant(x::AnyCuArray{T, 3}, M::Int) where {T} = cucirculant(size(x,1), M, T)
circulant(x::Array{T, 4}, M::Int) where {T} = circulant(size(x)[1:2], M, T)
circulant(x::Array{T, 3}, M::Int) where {T} = circulant(size(x,1), M, T)

function circulant_softmax!(Y::CuSparseMatrixCSR, X::CuSparseMatrixCSR=Y)
    V = reshape(X.nzVal, :, X.dims[1])
    U = reshape(Y.nzVal, size(V))
    NNlib.softmax!(U, V; dims=1)
    return Y
end

function circulant_softmax!(Y::SparseMatrixCSC, X::SparseMatrixCSC=Y)
    V = reshape(X.nzval, :, size(X, 2))
    U = reshape(Y.nzval, size(V))
    NNlib.softmax!(U, V; dims=1)
    return Y
end
circulant_softmax(X) = circulant_softmax!(copy(X), X)

