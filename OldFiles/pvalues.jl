using Distributions
1-cdf(Chisq(4),6.48)
1-cdf(Chisq(5),14.070577957326558)
1-cdf(Chisq(5),5.10529284806017)
1-cdf(Chisq(5),2.9674633675108444)
(1/0.996^4)-1
1-cdf(Chisq(4),6.14)
1-cdf(Chisq(4),6.476222211131761)
1-cdf(Chisq(4),6.476222211131761)
1-cdf(Chisq(4),6.476222211131761)
#Couples
1-cdf(Chisq(4),71.01463993941336)
1-cdf(Chisq(4),71.01463993941336)
1-cdf(Chisq(4),71.01463993941336)
1-cdf(Chisq(4),101.57907828136416)
##2APP
#PM
1-cdf(Chisq(150),17.87933251238829)
#TH
1-cdf(Chisq(150),299.13659166099404)
#IU
1-cdf(Chisq(7),9.046799601542176)
#Collective
1-cdf(Chisq(4),0.017779151532162054)
quantile(Chisq(4),0.95)

using CSV
P1=Array(CSV.read("/Users/nailkashaev/Dropbox/GitHub/ReplicationAK/Output_all/Appendix/F_1.0._0.975.csv"))
P2=Array(CSV.read("/Users/nailkashaev/Dropbox/GitHub/ReplicationAK/Output_all/Appendix/F_1.1._0.975.csv"))

crval=quantile(Chisq(5),.95) #critical values based on chi2(4)

I1=(P1[:,2].<crval)
I2=(P2[:,2].<crval)

println(P1[I1,:])
println(P2[I2,:])
