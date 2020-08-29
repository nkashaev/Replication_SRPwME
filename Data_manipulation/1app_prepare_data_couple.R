require(R.matlab)
## set data
dir="C:\\Users\\pegas\\Documents\\GitHub\\ReplicationAK"
##input files
irate.file=paste(dir,"\\","Data_all\\rawdata\\ir.mat",sep="")
expprice.file=paste(dir,"\\","Data_all\\rawdata\\ecpf_data.mat",sep="")
## output files
p.file=paste(dir,"\\","Data_all\\pcouple.csv",sep="")
cve.file=paste(dir,"\\","Data_all\\cvecouple.csv",sep="")
rv.file=paste(dir,"\\","Data_all\\rvcouple.csv",sep="")

irate.data=readMat(irate.file)
ecpf.data=readMat(expprice.file)
############################################################
## Data preparation
############################################################
## Erase missing values (1985 first 3 quarters have missing prices)
#ind.dum=is.nan(ecpf.data$hh.pricedata.all[,22])
##choose couples
ind.single=which(ecpf.data$hh.pricedata.all[,5]==1)
ind.dum=which(ecpf.data$hh.pricedata.all[ind.single,2]==1985)
hhid.miss=unique(ecpf.data$hh.pricedata.all[ind.single,][ind.dum,1])
hhid=unique(ecpf.data$hh.pricedata.all[ind.single,][-ind.dum,1])
miss.ind=c()
for (hh in hhid.miss){
	test.dum=which(hhid==hh)
	if (length(test.dum)>=1){
 		miss.ind=c(miss.ind,test.dum)
	}
}
hhid=hhid[-miss.ind]
#ind.dum2=which(ecpf.data$hh.pricedata.all[-ind.dum,22]=="NaN")
#hhid=unique(ecpf.data$hh.pricedata.all[!ind.dum,1][!ind.dum2])

n=length(hhid)
K=17
T=4
## initialization of arrays
## observed consumption
cve=array(NA,dim=c(n,T,K))
p=array(NA,dim=c(n,T,K))
rv=array(NA,dim=c(n,T))



## Prices data
## data description
#col1: hhid (household identifier)
#col2: year
#col3: quarter
#col4: t = {1,2,3,4}
#col5: couple (1 = yes, 0 = single)
#col6: all food and non-alc drinks price
#col7: all clothing price
#col8: (hserv) cleaning price
#col9: (hserv) nondur_article price
#col10: (hserv) hhservs price
#col11: (hserv) domservs price
#col12: (trans) other_trans price
#col13: (trans) pubtrans price
#col14: (trans) long-distance price
#col15: petrol price
#col16: (leisure) leisure1 price
#col17: (leisure) leisure2 price
#col18: (leisure) leisure3 price
#col19: (leisure) leisure4 price
#col20: (pserv) pserv1 price
#col21: (pserv) pserv2 price
#col22: food out price


##prices
for( i in 1:n){
       ind.dum=which(ecpf.data$hh.pricedata.all[,1]==hhid[i])

	for( k in 1:K){
  		for( t in 1:T){
	
   	 p[i,t,k]=ecpf.data$hh.pricedata.all[ind.dum,6:22][t,k] 

    }
  }
}

p=p/100

##expenditures
for( i in 1:n){
       ind.dum=which(ecpf.data$hh.expdata.all[,1]==hhid[i])

	for( k in 1:K){
  		for( t in 1:T){
	
   	 cve[i,t,k]=ecpf.data$hh.expdata.all[ind.dum,6:22][t,k] 

    }
  }
}

cve=cve/1e5
cve=cve/p
##interest rates

for( i in 1:n){
	
       ind.dum=which(ecpf.data$hh.expdata.all==hhid[i])
	yr=ecpf.data$hh.expdata.all[ind.dum,2][1]
	qtr=ecpf.data$hh.expdata.all[ind.dum,3][1]	
	ind.dum.1=which(irate.data$ir[,1]==yr)
      ind.dum.2=which(irate.data$ir[ind.dum.1,2]==qtr)
	ind.dum.3=ind.dum.1[ind.dum.2]-1
  	
	for( t in 1:T){
	r=irate.data$ir[,3][t+ind.dum.3]
	r= 1+ (10*(r-1))
      r=r^(1/4)
	r=r-1
   	rv[i,t]=r 	

    }
	rv[i,]=c(0,rv[i,1:(T-1)])
}


## Save

write.csv(p,file=p.file,row.names=FALSE,col.names=NA)
write.csv(cve,file=cve.file,row.names=FALSE,col.names=NA)
write.csv(rv,file=rv.file,row.names=FALSE,col.names=NA)

