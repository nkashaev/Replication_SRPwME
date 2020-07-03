#Here we need to list all Julia packages we need
tempdir1=@__DIR__
repdir=tempdir1[1:findfirst("ReplicationAK",tempdir1)[end]]
println(repdir)
println("Hello Victor")
#Pkg.add("sdsds")
