#Reading the results of determnistic tests and reporting average rejection rates
using CSV, DataFrames
using Distributions

tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
diroutput=repdir*"/Output_all/Appendix"

Result=DataFrame(zeros(2,2))
rename!(Result,Symbol.(["DGP","AverageRejRate"]))
Result[1,1]=1; Result[2,1]=2;

Result[1,2]=mean(CSV.read(diroutput*"/deter_null_theta0_0.8._n_2000.csv")[:,2])
Result[2,2]=mean(CSV.read(diroutput*"/deter_null_theta0_1.0._n_2000.csv")[:,2])

CSV.write(diroutput*"/deter_null_average_rejecton_rate.csv",Result)
