$WarningPreference = "SilentlyContinue"
$VerbosePreference="Continue"
Start-Transcript
##Singles


Measure-Command { D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\AK_FirstApp_singles_theta0_0975.jl"}
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\AK_FirstApp_singles_theta0_1.jl"
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\AK_FirstApp_singles_theta0_01.jl"
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\AK_FirstApp_singles_theta0_05.jl"
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\AK_FirstApp_singles_theta0_09.jl"

##Singles Recover Delta
Measure-Command { D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\AK_recover_ADF_997.jl"}
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\AK_recover_ADF_996.jl"
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\AK_recover_ADF_995.jl"
D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\singles\AK_recover_ADF_994.jl"

##Couples

D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\couples\AK_FirstApp_couples_theta0_01.jl"

Measure-Command { D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\couples\AK_FirstApp_couples_theta0_05.jl"}

D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\couples\AK_FirstApp_couples_theta0_09.jl"

D:\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\FirstApp\couples\AK_FirstApp_couples_theta0_1.jl"
