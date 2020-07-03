Readme file for the replication of the results in Section 8
Authors: Victor H. Aguiar and Nail Kashaev
Date: June 23, 2020
email: vaguiar@uwo.ca
This was done in a Windows 7 computer.
1. Install R 3.4.4
	1.1 Install revealedPrefs package from CRAN in R 3.4.4. (install.packages("revealedPrefs"))
	1.2. Unzip the revealedPrefsmod.zip package in Library for your R installation (next to where the revealedPrefs was installed, e.g. C:\Users\Nkashaev\Documents\R\R-3.4.4\library).
	1.3 Alternatively, instead of 1.2, you can compile from the source the revealedPrefsmod.tar file using Rtools. 
	This source file has the code for the functions that we use to simulate the draws of the true consumption and true price. 
	These are the functions:
	revealedPrefsmod.tar\revealedPrefsmodprog\src\ 
	simul.cpp: Original function in the Revprefs package. Simulates prices and quantities from a uniform distribution good by good.
 	simulprice.cpp: Modification of SimGarp. Takes prices as given and generates quantities.	
	simulpricewealth.cpp: Modification of SimGarp. Takes prices and expenditures and generates quantities that match them. 
	simulquantity.cpp: Modification of SimGarp. Takes quantities as given and generates prices.
	simulquantitywealth.cpp: Modification of SimGarp. Takes quantities and prices as given and generates prices. 

2. Install Julia v"1.4.2"
	1.1 Install the Julia Packages:  NLopt (this requires installing the NLopt optimizer libraries in Windows Version 2.4.1.), DataFrames, MathProgBase
	1.2. Install the Julia Package: RCall. This package must be installed After point 1 is done. This package seems to cause trouble in some systems with more than one R installation.
	Fix the ENV["R_HOME"] to the desired folder in those cases. 

4. Run AK_experimental_tremblind_hand.jl this reproduces the results in Section 8 about Trembling Hand Errors in Experimental Data. 
5. Run AK_experimental_price_misperception.jl this reproduces the results in Section 8 about Price Misperception Errors in Experimental Data. 
	




AK_Footnote_57_experimental_trembling_hand_reps_2970.jl Expected Execution Time: 10:08:10