block_dpa(q, k, v, windowsize) = windowed_dpa(q, k, v, windowsize)

function windowed_dpa(q::AbstractArray{T, N}, k::AbstractArray{T, N}, v::AbstractArray{T, N}, windowsize; kws...) where {T, N}
    qw = window(q, windowsize; kws...)
    kw = window(k, windowsize; kws...)
    vw = window(v, windowsize; kws...)

    yw, Pw = dense_dpa(
        reshape(qw, size(qw, 1), size(qw, 2), :),
        reshape(kw, size(kw, 1), size(kw, 2), :),
        reshape(vw, size(vw, 1), size(vw, 2), :))

    yw = reshape(yw, size(yw, 1), size(yw, 2), size(qw, 3), :)
    szy = (size(q)[1:N-2]..., size(v, N-1), size(q, N))

    divisor = unwindow(window(ones_like(v, szy), windowsize; kws...),
        szy, windowsize; kws...)

    y = unwindow(yw, szy, windowsize; kws...) ./ divisor
    P = reshape(Pw, size(Pw, 1), size(Pw, 2), :, size(q, N))
    return y, P
end
    
