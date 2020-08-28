### R 3.2.2
require(R.matlab)
## set data
dir="C:\\Users\\pegas\\Documents\\GitHub\\ReplicationAK\\datamanipulation\\rawdatafile"
##input files
Data.file=paste(dir,"\\","AllData.mat",sep="")
## output files
OutData.file=paste(dir,"\\","rationalitydata3goods.csv",sep="")

Data=readMat(Data.file)

OutData=cbind(1:dim(Data$AllData)[1],Data$AllData,round(100/Data$AllData[,6],digits=8),round(100/Data$AllData[,7],digits=8),round(100/Data$AllData[,8],digits=8))

write.csv(OutData,file=OutData.file,row.names=FALSE,col.names=NA)
