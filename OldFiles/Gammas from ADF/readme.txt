These files contain gammas that deliver the optimal values of the test statistic

Results1gamma=DataFrame(hcat(solvegamma,solvegamma))
CSV.write(diroutput*"/results_gamma_ADF_$avgdelta.csv",Results1gamma)
