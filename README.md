# Chemically-informed-data-driven-optimization-ChIDDO
Code used in the publication of Chemically-informed data-driven optimization (ChIDDO): Leveraging physical models and Bayesian learning to accelerate chemical research 

## Using python files
Each of the python files takes as input the two .csv files, "Alg_test.csv" and "Obj_test.csv".

"Alg_test.csv" is used to determine which algorithm to use (i.e. BO/ChIDDO, acquisition function, objective function). Multiple algorithms can be solved by adding another row to the file.

"Obj_test.csv" is used to define the parameters and other attribues of the objective function tested. The benchmark objective functions used in this publication are provided in this file, but others can be added by adding an additional row to the file.

## Using MATLAB files
MATLAB is being used as a more efficient way of solving electrochemical physics model systems. "Bayes_opt_auto.m" is the main script that will be run. The other functions in the MATLAB directory are auxillary functions that will be used in "Bayes_opt_auto.m". For this file, the parameters of the optimization are modified at the top of the script. The code will output files of the experimental points and a graph showing the most optimal experimental vs. number of experiments.
