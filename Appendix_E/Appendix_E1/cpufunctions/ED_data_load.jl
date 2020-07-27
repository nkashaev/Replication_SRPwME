function ED_data_load(dirdata,house)
  ## Data
  T=4; K=17;
  if house=="couples"
    n=2004
    #Prices
    p=reshape(Array(CSV.read(dirdata*"/pcouple.csv")),n,T,K)
    ## Consumption
    cve=reshape(Array(CSV.read(dirdata*"/cvecouple.csv")),n,T,K)
    ## Interest rates, +1.0 is needed following Adam et al. (2014) replication code.
    rv=Array(CSV.read(dirdata*"/rvcouple.csv")) .+ 1.0
  elseif house=="singles"
    n=185
    #Prices
    p=reshape(Array(CSV.read(dirdata*"/p.csv")),n,T,K)
    ## Consumption
    cve=reshape(Array(CSV.read(dirdata*"/cve.csv")),n,T,K)
    ## Interest rates
    rv=Array(CSV.read(dirdata*"/rv.csv")) .+ 1.0
  else
    println("Singles or Couples?")
  end
## Discounted prices
  rho=zeros(n,T,K)
  for i=1:n, t=1:T
      rho[i,t,:]=p[i,t,:]/prod(rv[i,1:t])
  end
  return rho, cve
end
