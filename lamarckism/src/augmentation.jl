# Includes random cropping and random cropout as well as a couple of workarounds to issues with augmentbatch!

Augmentor.applystepview(::FlipX, img::AbstractArray{T,N}, param) where {T,N} = Augmentor.indirect_view(img, (1:1:size(img,1), size(img,2):-1:1, (1:1:size(img,i) for i in 3:N)...))

Augmentor.applystepview(::FlipY, img::AbstractArray{T,N}, param) where {T,N} = Augmentor.indirect_view(img, (size(img,1):-1:1, (1:1:size(img,i) for i in 2:N)...))

pad(x, padsize) = PaddedView(0, x, size(x) .+ 2 .* padsize, padsize .+ 1)

RCrop(size::NTuple{N, <:Integer}, offs::NTuple{N, <:Integer}) where N = Either((1 .=> vcat(_RCrop(size,offs)...))...)
_RCrop(size, offs::NTuple{N, <:Integer}, oos::Integer...) where N = map(o -> vcat(_RCrop(size, Base.tail(offs), oos..., o)...), 0:first(offs))
_RCrop(size, ::Tuple{}, oos::Integer...) = (Crop(UnitRange.(oos .+ 1, size .+ oos)),)

function augitr(x, offs, seed=666)
    imgsize = size(x)
    # MLDatasets.CIFAR10 has horisontal axis in dimension 2. Check plot(CIFAR10.convert2image(augment(CIFAR10.traintensor(2), FlipX()))) if you don't believe me
    pipe = RCrop(imgsize[1:3], offs[1:3] .* 2) |> FlipY(0.5) |> Standardize(x) |> RCutout(8,8)

    # Set the seed for each thread so that all models see the exact same augmentation (given seed is the same). 
    sitr = i -> seeditr(i, seed % typemax(UInt32))
    return mapreduce(sitr, ∘, 1:Threads.nthreads()) ∘ (itr -> MapIterator(augfun(pipe, xx -> pad(xx, offs)), itr))
end
seeditr(tid, baseseed) = itr -> SeedIterator(itr; rng=Random.default_rng(tid), seed=tid + baseseed-1)

standardizeitr(x) = itr -> MapIterator(augfun(Standardize(x)), itr)

function Standardize(x)
    m = mean(x)
    s = std(x)
    return ConvertEltype(Float32) |> MapFun(x -> (x - m) / s)
end

augfun(pipe, wrap=identity) = x -> augmentbatch!(Array{Float32}(undef, size(x)...), wrap(x), pipe)


struct RCutout{N,I<:Tuple, R<:AbstractRNG} <: Augmentor.ImageOperation
    maxsize::I
    rng::R

    function RCutout{N}(maxsize::NTuple{N,<:Integer}, rng::R=Random.GLOBAL_RNG) where {N,R}
        new{N,typeof(maxsize),R}(maxsize, rng)
    end
end
RCutout(maxsize::Integer...; rng=Random.GLOBAL_RNG) = RCutout{length(maxsize)}(maxsize, rng)

@inline Augmentor.supports_eager(::Type{<:RCutout})      = false
@inline Augmentor.supports_affineview(::Type{<:RCutout}) = false
@inline Augmentor.supports_view(::Type{<:RCutout})       = false
@inline Augmentor.supports_stepview(::Type{<:RCutout})   = true

Augmentor.applystepview(op::RCutout, img::AbstractArray, param) = Augmentor.applyeager(op, img, param)
function Augmentor.applyeager(op::RCutout{N}, img::AbstractArray, param) where N
    start = rand.(Ref(op.rng), UnitRange.(1, size(img)[1:N] .- op.maxsize))
    csize = rand.(Ref(op.rng), UnitRange.(1, op.maxsize))
    cutout = UnitRange.(start, start .+ csize .- 1)
    inds = repeat(Any[Colon()], ndims(img))
    for (i, cu) in enumerate(cutout)
        if !isempty(cu)
            inds[i] = cu
        end
    end
    nimg = copy(img)
    nimg[inds...] .= randn(op.rng, csize)
    return nimg
end

Augmentor.toaffinemap(::FlipY, img::AbstractArray{T,3}) where T = Augmentor.recenter([-1. 0 0; 0 1. 0; 0 0 1], Augmentor.center(img))

Augmentor.toaffinemap(::FlipX, img::AbstractArray{T,3}) where T = Augmentor.recenter([1. 0 0; 0 -1. 0; 0 0 1], Augmentor.center(img))
