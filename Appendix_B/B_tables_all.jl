#Appendix B3, Table 2-4
using DataFrames, CSV
using Distributions

################################################################################
## Setting-up directory
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
appname="Appendix_B"
rootdir=repdir*"/"*appname
diroutput=repdir*"/Output_all"
dirdata=repdir*"/Data_all"
## Appendix B1, Table 2
##Loading the values of TS
RR_dgp_dettest=Array(CSV.read(diroutput*"/Appendix/B1_EDdettest_arr_dgp12.csv"))
dgp1=Array(CSV.read(diroutput*"/Appendix/B1_dgp1_chain_(0, 10000).sample_2000.csv"))
dgp2=Array(CSV.read(diroutput*"/Appendix/B1_dgp2_chain_(0, 10000).sample_2000.csv"))
##Computing average rejection rates
crval=quantile(Chisq(4),.95) #critical values based on chi2(4)
RR_dgp1=mean(dgp1[:,2].>crval)
RR_dgp2=mean(dgp2[:,2].>crval)
##Saving results
Table2=DataFrame(hcat(["DGP1" ;"DGP2"],[0.0; 0.0],[0.0; 0.0]))
rename!(Table2,Symbol.(["DGP","Deterministic test","Our methodology"]))
#Rejection rates measured in %
Table2[1,2]=round(RR_dgp_dettest[1,2] *100, digits=1)
Table2[1,3]=RR_dgp1 *100
Table2[2,2]=round(RR_dgp_dettest[2,2] *100, digits=1)
Table2[2,3]=RR_dgp2 *100
CSV.write(diroutput*"/Appendix/B1_Table2.csv",Table2)

## Appendix B2, Table 3
##Loading the values of TS
dgp3_2k=Array(CSV.read(diroutput*"/Appendix/B2_dgp3_chain_(0, 10000).sample_2000.csv"))
dgp3_3k=Array(CSV.read(diroutput*"/Appendix/B2_dgp3_chain_(0, 10000).sample_3000.csv"))
dgp4_2k=Array(CSV.read(diroutput*"/Appendix/B2_dgp4_chain_(0, 10000).sample_2000.csv"))
dgp4_3k=Array(CSV.read(diroutput*"/Appendix/B2_dgp4_chain_(0, 10000).sample_3000.csv"))
##Computing average rejection rates
crval=quantile(Chisq(4),.95) #critical values based on chi2(4)
RR_dgp3_2k=mean(dgp3_2k[:,2].>crval)
RR_dgp3_3k=mean(dgp3_3k[:,2].>crval)
RR_dgp4_2k=mean(dgp4_2k[:,2].>crval)
RR_dgp4_3k=mean(dgp4_3k[:,2].>crval)
##Saving results
Table3=DataFrame(hcat(["DGP3" "same" "different";"DGP4" "different" "different"],[0.0; 0.0],[0.0; 0.0]))
rename!(Table3,Symbol.(["DGP","prices","discount factors","n=2000","n=3000"]))
#Rejection rates measured in %
Table3[1,4]=round(RR_dgp3_2k *100, digits=1)
Table3[1,5]=round(RR_dgp3_3k *100, digits=1)
Table3[2,4]=round(RR_dgp4_2k *100, digits=1)
Table3[2,5]=round(RR_dgp4_3k *100, digits=1)
CSV.write(diroutput*"/Appendix/B2_Table3.csv",Table3)
## Appendix B3, Table 4
##Loading the values of TS
dgp2_10k=Array(CSV.read(diroutput*"/Appendix/B1_dgp2_chain_(0, 10000).sample_2000.csv"))
dgp2_5k=Array(CSV.read(diroutput*"/Appendix/B3_dgp2_chain_(0, 5000).sample_2000.csv"))
dgp3_10k=Array(CSV.read(diroutput*"/Appendix/B2_dgp3_chain_(0, 10000).sample_2000.csv"))
dgp3_5k=Array(CSV.read(diroutput*"/Appendix/B3_dgp3_chain_(0, 5000).sample_2000.csv"))
dgp4_10k=Array(CSV.read(diroutput*"/Appendix/B2_dgp4_chain_(0, 10000).sample_2000.csv"))
dgp4_5k=Array(CSV.read(diroutput*"/Appendix/B3_dgp4_chain_(0, 5000).sample_2000.csv"))
##Computing average rejection rates
crval=quantile(Chisq(4),.95) #critical values based on chi2(4)
RR_dgp2_10k=mean(dgp2_10k[:,2].>crval)
RR_dgp2_5k=mean(dgp2_5k[:,2].>crval)
RR_dgp3_10k=mean(dgp3_10k[:,2].>crval)
RR_dgp3_5k=mean(dgp3_5k[:,2].>crval)
RR_dgp4_10k=mean(dgp4_10k[:,2].>crval)
RR_dgp4_5k=mean(dgp4_5k[:,2].>crval)
##Saving results
Table4=DataFrame(hcat(["DGP2";"DGP3";"DGP4"],[0.0; 0.0; 0.0],[0.0; 0.0; 0.0]))
rename!(Table4,Symbol.(["DGP","cl10000","cl5000"]))
#Rejection rates measured in %
Table4[1,2]=round(RR_dgp2_10k *100, digits=1)
Table4[1,3]=round(RR_dgp2_5k *100, digits=1)
Table4[2,2]=round(RR_dgp3_10k *100, digits=1)
Table4[2,3]=round(RR_dgp3_5k *100, digits=1)
Table4[3,2]=round(RR_dgp4_10k *100, digits=1)
Table4[3,3]=round(RR_dgp4_5k *100, digits=1)
CSV.write(diroutput*"/Appendix/B3_Table4.csv",Table4)
