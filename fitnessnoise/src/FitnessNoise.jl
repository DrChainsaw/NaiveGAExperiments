module FitnessNoise

using Random
using Statistics
using Plots
using NaiveGAflux
pyplot()

export Cand, fitness, ga, gastat, EwmaCand, Ma, Resamp, LowBar, HighBar, Mutating

const rng_default = NaiveGAflux.rng_default;

include("evolve.jl")


fitness_true(c::AbstractCandidate) = fitness_true(base(c))
nvar(c::AbstractCandidate) = nvar(base(c))
speed(c::AbstractCandidate) = speed(base(c))

ff(x) = 12

struct Cand <: AbstractCandidate
    fitness::Float64
    gen::Int
    nvar::Float64
    speed::Float64
end
Cand(finalacc, speed, nvar) = Cand(finalacc, 1, nvar, speed)
Cand() = Cand(rand(), mut(0.05, 0.01), mut(0.05, 0.01))

NaiveGAflux.fitness(c::Cand) = min(1.0, fitness_noise(c) + fitness_gen(c))

fitness_noise(c::Cand) = c.nvar * randn()
fitness_gen(c::Cand) = c.fitness * c.gen / (c.gen + 1/(c.speed))
fitness_true(c::Cand) = c.fitness

nvar(c::Cand) = c.nvar
speed(c::Cand) = c.speed

# Strong bias towards making models worse after mutation
mutate(c::Cand) = Cand(fmut(c.fitness, 0.1, 0.8), c.gen, mut(c.nvar, 0.05), mut(c.speed, 0.1))
nextgen(c::Cand) = Cand(c.fitness, c.gen+1, c.nvar, c.speed)

mut(n, mr, b=0.5) = n * (1 + mr * (rand() - b))

# Negative bias approaches 1.0 as fitness approaches 1.0 meaning it becomes harder and harder to improve the fitness
fmut(f, mr, b) = mut(f, mr, b * (1 - f^20) + f^20)

mutable struct CacheCand <: AbstractCandidate
    fitness
    c::AbstractCandidate
end
CacheCand() = CacheCand(Cand())
CacheCand(c) = CacheCand(missing, c)
base(c::CacheCand) = c.c


function NaiveGAflux.fitness(c::CacheCand)
    if ismissing(c.fitness)
        c.fitness = fitness(c.c)
    end
    return c.fitness
end

nextgen(c::CacheCand) = return CacheCand(nextgen(c.c))

mutate(c::CacheCand) = CacheCand(c.fitness, mutate(c.c))


mutable struct EwmaCand <: AbstractCandidate
    fitness
    α::Float64
    c::AbstractCandidate
end
EwmaCand(;α = 0.5, c=Cand()) = EwmaCand(missing, α, c)
base(c::EwmaCand) = c.c

function NaiveGAflux.fitness(c::EwmaCand)
    if ismissing(c.fitness)
        c.fitness = fitness(c.c)
    else
        c.fitness = (1 - c.α) * c.fitness + c.α * fitness(c.c)
    end
    return c.fitness
end

mutate(c::EwmaCand) = EwmaCand(c.fitness, c.α, mutate(c.c))
nextgen(c::EwmaCand) = EwmaCand(c.fitness, c.α, nextgen(c.c))

mutable struct LowBar <: AbstractCandidate
    last::Float64
    worst::Float64
    c::AbstractCandidate
end
LowBar() = LowBar(Cand())
LowBar(c) = LowBar(0.0, 0.0, c)
base(c::LowBar) = c.c

function NaiveGAflux.fitness(c::LowBar)
    fraw = fitness(c.c)
    Δ = min(0, fraw - c.last)
    α = 0.05
    c.worst = (1 - α) * c.worst + α * abs(Δ)
    c.last = fraw
    return fraw - abs(c.worst)
end


mutate(c::LowBar) = LowBar(c.last, c.worst, mutate(c.c))
nextgen(c::LowBar) = LowBar(c.last, c.worst, nextgen(c.c))

mutable struct LowBarEwma <: AbstractCandidate
    last::Float64
    worst::Float64
    fitness
    α::Float64
    c::AbstractCandidate
end
LowBarEwma(α=0.01) = LowBarEwma(Cand(), α)
LowBarEwma(c, α) = LowBarEwma(0.0,0.0,missing,α, c)
base(c::LowBarEwma) = c.c

function NaiveGAflux.fitness(c::LowBarEwma)
    fraw = fitness(c.c)
    if ismissing(c.fitness)
        c.fitness = fraw
    else
        c.fitness = (1 - c.α) * c.fitness + c.α * fraw
    end
    Δf = fraw - c.fitness
    Δ = min(0, fraw - c.last)
    #c.worst = min(c.worst, Δ)
    α = c.α
    c.worst = (1 - α) * c.worst + α * abs(Δ)
    c.last = fraw
    return c.fitness - abs(c.worst)
end


mutate(c::LowBarEwma) = LowBarEwma(c.last, c.worst, c.fitness, c.α, mutate(c.c))
nextgen(c::LowBarEwma) = LowBarEwma(c.last, c.worst, c.fitness, c.α, nextgen(c.c))

mutable struct HighBar <: AbstractCandidate
    best::Float64
    c::AbstractCandidate
end
HighBar(c=Cand()) = HighBar(0.0, c)
base(c::HighBar) = c.c

function NaiveGAflux.fitness(c::HighBar)
    fraw = fitness(c.c)
    c.best = max(c.best, fraw)
    return fraw
end

mutate(c::HighBar) = HighBar(c.best, mutate(c.c))
nextgen(c::HighBar) = HighBar(c.best, nextgen(c.c))

mutable struct Resamp <: AbstractCandidate
    n
    c::AbstractCandidate
end
Resamp(n=100) = Resamp(n, Cand())
base(c::Resamp) = c.c

function NaiveGAflux.fitness(c::Resamp)
    fsum = mapreduce(i -> fitness(c.c), +, 1:c.n)
    return fsum / c.n
end

mutate(c::Resamp) = Resamp(c.n, mutate(c.c))
nextgen(c::Resamp) = Resamp(c.n, nextgen(c.c))

struct Ma <: AbstractCandidate
    fw::Vector{Float64}
    c::AbstractCandidate
end
Ma(fw = zeros(2)) = Ma(fw, Cand())
base(c::Ma) = c.c

function NaiveGAflux.fitness(c::Ma)
    c.fw[1:end-1] = c.fw[2:end]
    c.fw[end] = fitness(c.c)
    return mean(c.fw)
end

mutate(c::Ma) = Ma(copy(c.fw), mutate(c.c))
nextgen(c::Ma) = Ma(copy(c.fw), nextgen(c.c))


struct Mutating{T} <: AbstractCandidate
    c::T
end
base(c::Mutating) = c.c

NaiveGAflux.fitness(c::Mutating) = fitness(c.c)

nextgen(c::Mutating) = Mutating(nextgen(c.c))
mutate(c::Mutating{EwmaCand}) = Mutating(EwmaCand(c.c.fitness, mut(c.c.α, 0.01), mutate(c.c.c)))

struct Hist
    lab::String
    f::Vector{Float64}
    t::Vector{Float64}
    n::Vector{Float64}
    s::Vector{Float64}
end
Hist(lab) = Hist(lab,Float64[],Float64[],Float64[],Float64[])
function (p::Hist)(c::AbstractCandidate)
    push!(p.f, fitness(c))
    push!(p.t, fitness_true(c))
    push!(p.n, nvar(c))
    push!(p.s, speed(c))
    return p
end

function (p::Hist)(pop::Vector{<:AbstractCandidate})
    push!(p.f, mean(map(fitness, pop)))
    push!(p.t, mean(map(fitness_true, pop)))
    push!(p.n, mean(map(nvar, pop)))
    push!(p.s, mean(map(speed,pop)))
    return p
end

function Plots.plot(h::Hist, pf=plot, pt=plot, pn=plot, ps=plot)
    return pf(h.f; label=h.lab, title="Evaluated accuracy"),
    pt(h.t; label=h.lab, title="True accuracy"),
    pn(h.n; label=h.lab, title="Noise variance"),
    ps(h.s; label=h.lab, title="Convergence speed")
end

Plots.plot!(h::Hist, pf, pt, pn, ps) = plot(h, plt!(pf), plt!(pt), plt!(pn), plt!(ps))
plt!(p) = (a...;kw...) -> plot!(p, a...;kw...)

function evostrategy(ps, ne)
    elite = EliteSelection(ne)
    evolve = TournamentSelection(ps - ne, 2, 1.0, EvolveCandidates(mutate))
    combine = CombinedEvolution(elite, evolve)
    switch = combine
    return AfterEvolution(switch, p -> nextgen.(p))
end


function ga(ps= 128, ng=200, seed=0; cfun = () -> CacheCand(EwmaCand(α = 0.5)))
    pyplot()
    Random.seed!(rng_default, seed)
    pop = [cfun() for _ in 1:ps]
    estrat = evostrategy(ps, 2)

    hb = Hist("Best")
    ht = Hist("Best true")
    hm = Hist("Mean")

    for i in 1:ng
        popstat(pop, hb, ht, hm)
        pop = evolve!(estrat, pop)
    end
    return plotpop(hb,ht,hm)

end

function gastat(ps=128, ng=1000, ns = 200; fs = (CacheCand,))
    plt = plot(xlabel="True accuracy of final best candidate", ylabel="Empirical CDF")
    for cfun in fs
        bf = Vector{Float64}(undef, ns)
        Threads.@threads for seed in 1:ns
            i = seed
            Random.seed!(rng_default, seed)
            pop = [cfun() for _ in 1:ps]
            estrat = evostrategy(ps, 2)

            for i in 1:ng
                pop = evolve!(estrat, pop)
            end
            bf[i] = fitness_true(pop[1])
        end
        plot!(plt, sort(bf), range(0, 100, length=length(bf)), label = string(typeof(base(cfun()))));
    end
    return plt
end

bestfit_true(pop) = partialsort(pop, 1, by = fitness_true, rev=true)

function popstat(pop, hb, ht, hm)
    hb(first(pop))
    ht(bestfit_true(pop))
    hm(pop)
end

function plotpop(hb,ht,hm)
    ps = plot(hb)
    plot!(ht, ps...)
    plot!(hm, ps...)
    plot(ps...)
end

end
