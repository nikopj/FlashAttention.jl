@inline block_fa(q, k, v, windowsize; pad=0) = windowed_fa(q, k, v, windowsize; stride=windowsize, pad=pad)

function windowed_fa(q::AbstractArray{T, N}, k::AbstractArray{T, N}, v::AbstractArray{T, N}, windowsize; kws...) where {T, N}
    qw = window(q, windowsize; kws...)
    kw = window(k, windowsize; kws...)
    vw = window(v, windowsize; kws...)

    yw, lw, mw = dense_fa(
        reshape(qw, size(qw, 1), size(qw, 2), :),
        reshape(kw, size(kw, 1), size(kw, 2), :),
        reshape(vw, size(vw, 1), size(vw, 2), :))

    yw = reshape(yw, size(yw, 1), size(yw, 2), size(qw, 3), :)
    szy = (size(q)[1:N-2]..., size(v, N-1), size(q, N))

    divisor = unwindow(window(ones_like(v, szy), windowsize; kws...),
        szy, windowsize; kws...)

    y = unwindow(yw, szy, windowsize; kws...) ./ divisor
    l = reshape(lw, size(lw, 1), size(lw, 2), :, size(q, N))
    m = reshape(mw, size(mw, 1), size(mw, 2), :, size(q, N))
    return y, l, m
end
