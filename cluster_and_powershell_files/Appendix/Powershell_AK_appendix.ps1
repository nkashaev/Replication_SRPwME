﻿$WarningPreference = "SilentlyContinue"
$VerbosePreference="Continue"
Start-Transcript
##Singles
Write-Host "COLLECTIVE"
#D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\Appendix_E\Appendix_E2\AK_collective.jl"
Measure-Command { D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\Appendix_E\Appendix_E2\AK_collective.jl"}

Write-Host "INCOME UNCERTAINTY"
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\Appendix_E\Appendix_E1\AK_IncomeUncertainty.jl"
Measure-Command { D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\Appendix_E\Appendix_E1\AK_IncomeUncertainty.jl"}

##Singles Recover Delta

