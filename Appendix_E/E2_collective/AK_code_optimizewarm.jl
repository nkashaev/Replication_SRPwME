model=nothing
modelm=nothing
chainMcu=nothing
GC.gc()
modelm=JuMP.Model(with_optimizer(Ipopt.Optimizer))
@variable(modelm, -10e300 <= gammaj[1:dg] <= 10e300)

@NLobjective(modelm, Min, sum(log(1+sum(exp(sum(chainMnew[id,t,j]*gammaj[t] for t in 1:dg)) for j in 1:nfast)/nfast) for id in 1:n)/n )


chainM

JuMP.optimize!(modelm)

gammastep5=zeros(dg)
for d=1:dg
    gammastep5[d]=JuMP.value(gammaj[d])
end
