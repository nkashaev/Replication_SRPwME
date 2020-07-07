########################################################################################################
@everywhere function new_delta(;d=d,vsim=vsim,cvesim=cvesim,rho=rho)
    d=d
    dmin=d
    dmax=1

    for t=2:T

        for s=1:T

            numer=(sum(rho[t,:].*(cvesim[t,:]-cvesim[s,:])))
            denom=(vsim[t,1]-vsim[s,1])


            if denom>0
#                println(s,t)
                val1=maximum([0,(numer/denom)])^(1/(t-1))
                dmin=maximum([dmin,val1])

            end
            if denom<0
#                println(s,t)
                val1=maximum([0,(numer/denom)])^(1/(t-1))
                dmax=minimum([dmax,val1])
            end
        end
    end
    maximum([dmin,d]),minimum([dmax,1])

end
