### R 3.2.2
require(R.matlab)
## set data
dir="C:\\Users\\pegas\\Documents\\GitHub\\ReplicationAK\\datamanipulation\\rawdatafile"
##input files
Data.file=paste(dir,"\\","AllData.mat",sep="")
## output files
OutData.file=paste(dir,"\\","rationalitydata3goods.csv",sep="")

Data=readMat(Data.file)

Data$AllData