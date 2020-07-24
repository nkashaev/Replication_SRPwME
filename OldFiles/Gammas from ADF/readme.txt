Date: July 23, 2020
These files contain gammas that deliver the optimal values of the test statistic
for computing the confidence set for the average discount factor.

Results1gamma=DataFrame(hcat(solvegamma,solvegamma))
CSV.write(diroutput*"/results_gamma_ADF_$avgdelta.csv",Results1gamma)
