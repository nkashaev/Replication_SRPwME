


for (($ki = 1); $ki -lt 11+1; $ki++)
{
    for (($ri = 1); $ri -lt 41+1; $ri++)
    {
        "`$ki:$ki"
        "`$ri:$ri"
        C:\Users\vaguiar\AppData\Local\Julia-1.1.1\bin\julia.exe "C:\Users\vaguiar\Documents\GitHub\ReplicationAK\Appendix_F\F_main_shell.jl" $ki $ri
    }
}
