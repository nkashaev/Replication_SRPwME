require(R.matlab)
## set data
dir="C:\\Users\\pegas\\Documents\\GitHub\\ReplicationAK"
##input files
Data.file=paste(dir,"\\","Data_all\\rawdata\\AllData.mat",sep="")
## output files
OutData.file=paste(dir,"\\","Data_all\\rationalitydata3goods.csv",sep="")

Data=readMat(Data.file)

OutData=cbind(1:dim(Data$AllData)[1],Data$AllData,round(100/Data$AllData[,6],digits=9),round(100/Data$AllData[,7],digits=9),round(100/Data$AllData[,8],digits=9))
colnames(OutData)=c("","id","obs","x","y","z","xa","ya","za","px","py","pz")

write.csv(OutData,file=OutData.file,row.names=FALSE)
