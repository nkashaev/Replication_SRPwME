#jumpfun
@everywhere function new_VC_long(VC,P)
  # given the initial matrix VC and prices P
  # the function samples from the polytope uniformly

  thetamin=-10^6 #initial upperbound
  thetamax=10^6 #initial lowerbound
  #Generating random direction
  #v=next!(s)
  #dVC=reshape(v./norm(v),size(VC))
  v=randn(size(VC))
  dVC=v./norm(v)
  # #Box constraints

  #box constraints do not include v numbers, but included without loss of generality
    for i=1:size(VC,2)
      for j=1:size(VC,1)
          if dVC[j,i]<0
              thetamax=minimum([thetamax,-(VC[j,i])/(dVC[j,i])])
          else
              thetamin=maximum([thetamin,-(VC[j,i])/(dVC[j,i])])
          end
      end
  end
  #Afriat constraints
  for i=1:size(VC,1)
      for j=1:size(VC,1)
          if j!=i
              output_denom1 =  dot(vcat(-1.0, P[j,:]), dVC[j,:] .- dVC[i,:])
              output_num1   = -dot(vcat(-1.0, P[j,:]), VC[j,:] .- VC[i,:])

              if output_denom1>0
                  thetamax=minimum([thetamax, (output_num1)/(output_denom1)])
              else
                  thetamin=maximum([thetamin, (output_num1)/(output_denom1)])
              end
          end
      end
  end

    newdir=(rand()*(thetamax-thetamin)+thetamin)
    return VC[:,1] .+ newdir.* dVC[:,1], VC[:,2:end] .+ newdir.* dVC[:,2:end]
end;
