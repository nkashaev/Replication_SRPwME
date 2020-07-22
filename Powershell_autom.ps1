$WarningPreference = "SilentlyContinue"
$VerbosePreference="Continue"
Start-Transcript
##Couples 

D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\couples\AK_FirstApp_couples_theta0_01.jl"

Measure-Command { D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\couples\AK_FirstApp_couples_theta0_05.jl"}

D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\couples\AK_FirstApp_couples_theta0_09.jl"

D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\couples\AK_FirstApp_couples_theta0_1.jl"

