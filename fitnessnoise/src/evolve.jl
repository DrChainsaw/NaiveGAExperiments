mutable struct SwitchEvolution <: AbstractEvolution
    n::Int
    cnt::Int
    evo1::AbstractEvolution
    evo2::AbstractEvolution
end
SwitchEvolution(n, evo1, evo2) = SwitchEvolution(n, 0, evo1, evo2)
function NaiveGAflux.evolve!(e::SwitchEvolution, pop)
    evo = e.cnt % e.n == 0 ? e.evo1 :  e.evo2
    e.cnt += 1
    return evolve!(evo, pop)
end
