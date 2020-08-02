$WarningPreference = "SilentlyContinue"
$VerbosePreference="Continue"
Start-Transcript
##Singles
Write-Host "SINGLES"

Measure-Command { D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\1App_singles_main_0.975.jl"}
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\1App_singles_main_0.1.jl"
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\1App_singles_main_0.5.jl"
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\1App_singles_main_0.9.jl"
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\1App_singles_main_1.0.jl"

Write-Host "SINGLES RECOVER"

##Singles Recover Delta
Measure-Command { D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\1App_singles_ADF_0.996.jl"}
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\1App_singles_ADF_0.995.jl"
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\1App_singles_ADF_0.994.jl"


##Couples
Write-Host "COUPLES"

Measure-Command { D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\couples\1App_couples_main_0.1.jl"}
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\couples\1App_couples_main_0.5.jl"
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\couples\1App_couples_main_0.9.jl"
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\couples\1App_couples_main_1.0.jl"
