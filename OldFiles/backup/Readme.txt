Authors: Victor H. Aguiar and Nail Kashaev
Date: June 23, 2020
email: vaguiar@uwo.ca
This was done in a Windows 7 computer.
1. Install R 3.4.4
	1.1 Install revealedPrefs package from CRAN in R 3.4.4. (install.packages("revealedPrefs"))
	1.2. Unzip the revealedPrefsmod.zip package in Library for your R installation (next to where the revealedPrefs was installed, e.g. C:\Users\Nkashaev\Documents\R\R-3.4.4\library).
	1.3 Alternatively, instead of 1.2, you can compile from the source the revealedPrefsmod.tar file using Rtools. 
	This source file has the code for the functions that I use to simulate the draws of the true consumption and true price. 
	These are the functions:
	export(simGarp).- Original function in the Revprefs package. Simulates prices and quantities from a uniform distribution good by good. This is very fast because it uses DFS methods that are recursive. Recall that the distribution of the guesses for the Montecarlo are irrelevant as long as it samples from all the space. Does not impose budget constraints or uses data.
	export(simGarpPrice).- My modification of SimGarp. Takes prices as given and generates quantities.	
	export(simGarpPriceWealth).- My modification of SimGarp. Takes prices and expenditures and generates quantities that match them. 
	export(simGarpQuant).- My modification of SimGarp. Takes quantities as given and generates prices.
	export(simGarpQuantWealth).- My modification of SimGarp. Takes quantities and prices as given and generates  prices. 

2. Install Julia V.0.5.1 
	1.1 Install the Julia Packages:  NLopt (this requires installing the NLopt optimizer libraries in Windows Version 2.4.1.),  DataFrames, MathProgBase
	1.2. Install the Julia Package: RCall. This package must be installed After point 1 is done. This package seems to cause trouble in some systems with more than one R installation.
	I added in the code one additional lines to fix the enviorenment to the desired intallation and the related library. 
3. Performance. The NLopt is not the best optimizer out there. For high-dimensions it can get a bit slow. In principle one can replace it by a better one, maybe paid. I tried Knitro, not a great option. 
3.1 The simulation step is fast due to the Revprefs package that I modified myself for this application. I have tried with pure Julia implementations and they cannot reach the efficiency of this package. The reason is that most functions are RcppArmadillo. 
3.2 In principle one can replace the guessfun and jumpfun functions for the Montecarlo for another sampler. 

4. Run AK_experimental_main.jl 
	4.1 Be careful to choose the number of processors, the number of time periods you want to consider (for this demo I suggest trying first small T=4), tolrance levels and which model to run. 
	4.2 There are 2 options: (i) Default: uncommented is the trembling hand case, (ii) the commented code is for the misperception in price case. Check the paper for details. 
	

