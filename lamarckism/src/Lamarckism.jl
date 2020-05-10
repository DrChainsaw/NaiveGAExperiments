module Lamarckism

using NaiveGAflux
using MLDatasets: CIFAR10

using Augmentor
using PaddedViews
using ONNXmutable

using Random
using Statistics


include("augmentation.jl")

(::NaiveGAflux.GlobalPool{MaxPool})(pp::ONNXmutable.AbstractProbe) = ONNXmutable.globalmaxpool(pp, y -> dropdims(y; dims=(1,2)))
(::NaiveGAflux.GlobalPool{MeanPool})(pp::ONNXmutable.AbstractProbe) = ONNXmutable.globalmeanpool(pp, y -> dropdims(y; dims=(1,2)))


struct BoostPars{T, V}
    lrboost::T
    ps::V
end

function Flux.Optimise.apply!(o::BoostPars, x, Δ)
    if x in o.ps
        @. Δ = Δ * o.lrboost
    end
    return Δ
end

struct BoostNotPars{T, V}
    lrboost::T
    ps::V
end

function Flux.Optimise.apply!(o::BoostNotPars, x, Δ)
    if x ∉ o.ps
        @. Δ = Δ * o.lrboost
    end
    return Δ
end

# Experiments in blogpost don't use biases, probably for speed reasons. This will not give the same speed boost as all it does it that we don't update the bias.
Flux.trainable(l::Conv) = (l.weight,)
Flux.trainable(l::Dense) = (l.W,)

# Change momentum to work the same way as in PyTorch in an attempt to get the same performance
function Flux.Optimise.apply!(o::Momentum, x, Δ)
  η, ρ = o.eta, o.rho
  v = get!(o.velocity, x, zero(x))::typeof(x)
  @. v = ρ * v - Δ
  @. Δ = - η * v
end

function train(model, ne, optf, bs=512)

    model = model |> gpu

    loss(x,y) = Flux.logitcrossentropy(model(x), y)
    iter = trainiter(CIFAR10.traindata()..., bs)
    accmeas = AccuracyFitness(testiter(CIFAR10.testdata()..., bs))

    accuracy = []
    for epoch in 1:ne
        opt = optf(epoch)
        Flux.train!(loss, params(model), iter, opt)

        push!(accuracy, fitness(accmeas, model))

        println("Epoch $(epoch),\tlr: $(round(lr(opt), sigdigits=3)),\ttest accuracy: $(accuracy[end])")
    end

    return accuracy, cpu(model)
end

function trainiter(x,y, bs, seed=123)
    rng() = MersenneTwister(seed)

    return GpuIterator(zip(ShuffleIterator(copy(x), bs, rng()) |> augitr(x, (4,4,0,0), 666), Flux.onehotbatch(ShuffleIterator(y, bs, rng()), 0:9)))
end

testiter(x,y, bs) = GpuIterator(zip(BatchIterator(x, bs) |> standardizeitr(x), Flux.onehotbatch(BatchIterator(y, bs), 0:9)))

standardize(x) = (x .- mean(x)) ./ std(x)

lrsched(ne) = Float32.(vcat(range(0.1, 0.4;length=4), range(0.375, 0;length=ne-4)))

lr(o) = Flux.Optimise.apply!(o, [0.0], [1.0])[1]


function backbone()
    function cbr(vin, outsize, maybepool = v -> mutable("maxpool$outsize", MaxPool((2,2); stride=(2,2)), v))
        c = mutable("conv$outsize", Conv((3,3), nout(vin) => outsize;pad=1, init=he_normal), vin)
        b = mutable("bn$outsize", BatchNorm(nout(c), relu), c)
        return maybepool(b)
    end

    v0 = inputvertex("input", 3, FluxConv{2}())
    prep = cbr(v0, 64, identity)
    l1 = cbr(prep, 128)
    l2 = cbr(l1, 256)
    l3 = cbr(l2, 512)
    gp = GlobalPoolSpace(MaxPool)("globpool", l3)
    out = mutable("linear", Dense(nout(gp), 10;initW=uniform), gp)
    scal = invariantvertex(x -> 0.125f0 .* x, out)
    return CompGraph(v0, scal)
end

function resnet()
    function cbr(vin, outsize, maybepool = v -> mutable("maxpool$outsize", MaxPool((2,2); stride=(2,2)), v))
        c = mutable("conv$outsize", Conv((3,3), nout(vin) => outsize;pad=1, init=he_normal), vin)
        b = mutable("bn$outsize", BatchNorm(nout(c), relu), c)
        return maybepool(b)
    end

    function resblock(in)
        res1 = cbr(in, nout(in), identity)
        res2 = cbr(res1, nout(in), identity)
        return in + res2
    end

    v0 = inputvertex("input", 3, FluxConv{2}())
    prep = cbr(v0, 64, identity)
    l1 = resblock(cbr(prep, 128))
    l2 = cbr(l1, 256)
    l3 = resblock(cbr(l2, 512))
    gp = GlobalPoolSpace(MaxPool)("globpool", l3)
    out = mutable("linear", Dense(nout(gp), 10;initW=uniform), gp)
    scal = invariantvertex(x -> 0.125f0 .* x, out)
    return CompGraph(v0, scal)
end

pretrained(fname) = CompGraph(fname)


NaiveNASflux.layer(f) = f
NaiveGAflux.resinitW(wi::Union{NaiveGAflux.IdentityWeightInit, NaiveGAflux.PartialIdentityWeightInit}) = wi

function addres_fixed(g, vs; wi=IdentityWeightInit())
    conv2d = VertexSpace(NamedLayerSpace("conv2d", ConvSpace2D(128:128, [identity], [3])))
    bn = VertexSpace(NamedLayerSpace("batchnorm", BatchNormSpace([relu])))

    convbn = RepeatArchSpace(ListArchSpace(conv2d, bn),2)
    resspace = ResidualArchSpace(convbn)
    scalespace = FunctionSpace(x -> 0.5f0 * x; namesuff="scale")
    add_conv = AddVertexMutation(ListArchSpace(resspace, scalespace), wi)

    vsorg = vertices(g)
    add_conv.(vsorg[vs])
    vsnew = setdiff(vertices(g), vsorg)

    psnew = params(layer.(vsnew))

    return g, psnew.order
end

function addres_learnable(g, vs; λ=0.5, wi = IdentityWeightInit())

    # Turns out this is alot faster than Flux.Diagonal...
    diag(size, val) = Conv((1,1), size=>size; init=(args...) -> Float32(val) .* idmapping(args...))

    cinit = NaiveGAflux.convinitW(wi).init
    conv(size) = Conv((3,3), size => size, identity;init=cinit, pad=(1,1))
    bn(size) = BatchNorm(size, relu)
    function resblock(v, λ)
        c1 = mutable(conv(nout(v)), v)
        b1 = mutable(bn(nout(c1)), c1)
        c2 = mutable(conv(nout(b1)), b1)
        b2 = mutable(bn(nout(c2)), c2)
        rscale = mutable(diag(nout(b2), 1 - λ), b2)

        oscale = mutable(diag(nout(v), λ), v)

        return rscale + oscale
    end

    vsorg = vertices(g)
    insert!.(vsorg[vs], v -> resblock(v, λ))
    vsnew = setdiff(vertices(g), vsorg)

    psnew = params(layer.(vsnew))

    return g, psnew.order
end

struct BlendWeightInit{T,V,B} <: NaiveGAflux.AbstractWeightInit
    w1::T
    w2::V
    blend::B
end

function NaiveGAflux.convinitW(wi::BlendWeightInit)
    b1 = wi.blend
    b2 = 1 - wi.blend
    w1 = NaiveGAflux.convinitW(wi.w1).init
    w2 = NaiveGAflux.convinitW(wi.w2).init
    return (init=(args...) -> b1 * w1(args...) + b2 * w2(args...),)
end

struct HeWeightInit <: NaiveGAflux.AbstractWeightInit end
NaiveGAflux.convinitW(wi::HeWeightInit) = (init=he_normal,)

nfanin() = 1
nfanin(n) = 1 #A vector is treated as a n×1 matrix
nfanin(n_out, n_in) = n_in #In case of Dense kernels: arranged as matrices
nfanin(dims...) = prod(dims[1:end-2]) .* dims[end-1] #In case of convolution kernels
he_normal(dims...) = randn(Float32, dims...) .* sqrt(2.0f0 / nfanin(dims...))

uniform(dims...) = (rand(Float32, dims...) .- 0.5) .* 2f0 / Float32(sqrt(nfanin(dims...)))


end
