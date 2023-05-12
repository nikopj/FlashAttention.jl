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

function circulant(N::Int, M::Int, Tv=Float64, Ti=Int64) 
    rowval = (Ti∘first∘cartesian_circulant).(1:N*M, N, M) 
    colptr = 1 .+ M .* collect(0:N) .|> Ti
    return SparseMatrixCSC{Tv, Ti}(N, N, colptr, rowval, ones(Tv, N*M))
end

