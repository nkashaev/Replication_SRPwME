## R version 3.6.1 (2019-07-05)
##Author: vhaguiar@gmail.com
library("tikzDevice")
require("ggplot2")
library("hrbrthemes")

filefigure="C:/Users/pegas/Documents/GitHub/ReplicationAK/Output_all/Appendix/figures"

theta0str="0.975"
##Petrol good 10
targetgood="10"
simulations=11
results <- vector("list", simulations) 
multiplier=c("1.0","1.01","1.02","1.03","1.04","1.05","1.06","1.07","1.08","1.09","1.1")


for (i in 1:(simulations)){
  file.alias=paste("C:/Users/pegas/Documents/GitHub/ReplicationAK/Output_all/Appendix/counter.good_10._price_10._multiplier_",multiplier[i],"._cuda_start.0.04.end.0.06.theta0.",theta0str,".csv",sep="")
  results[[i]]<- read.csv(file.alias)
}

crit.value=qchisq(.95,df=5)

lbound=rep(0,simulations)
ubound=rep(0,simulations)

for (i in 1:simulations){
  inddum=which(results[[i]][,2]==0)
  vecres=results[[i]][,2]
  vecres[inddum]=Inf
  passind=which(vecres<crit.value)
  lbound[i]=results[[i]][min(passind),1]
  ubound[i]=results[[i]][max(passind),1]
}

vec.mult=as.numeric(multiplier)



price=c(vec.mult,vec.mult)
shares=c(ubound,lbound)
type=c(rep("upper bound",simulations),rep("lower bound",simulations))
figdf=data.frame(type,price,shares)

fig1=ggplot(figdf, aes(x=price, y=shares, group=type)) +
  geom_line(aes(linetype=type))+
  geom_point()+
  scale_linetype_manual(values=c("twodash", "dotted"))
fig1
fig1 +labs(x="price multiplier")+ theme_minimal()

pdf(paste(filefigure,"\\good_10_price_10_2020_01_08_horizontal_",theta0str,"_minimal.pdf",sep=""),width =10,height =2)
myplot <- fig1 +labs(x="price multiplier") + theme_minimal()
print(myplot)
dev.off()

tikz(paste(filefigure,"\\good_10_price_10_2020_01_08_horizontal_",theta0str,"_minimal.tex",sep=""),width =10,height =2)
myplot <- fig1+ labs(x="price multiplier")+ theme_minimal()
print(myplot)
dev.off()

