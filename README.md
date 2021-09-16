# Chemically-informed-data-driven-optimization-ChIDDO
Code used in the publication of Chemically-informed data-driven optimization (ChIDDO): Leveraging physical models and Bayesian learning to accelerate chemical research 

## Using python files
Each of the python files takes as input the two .csv files, "Alg_test.csv" and "Obj_test.csv".

"Alg_test.csv" is used to determine which algorithm to use (i.e. BO/ChIDDO, acquisition function, objective function). Multiple algorithms can be solved by adding another row to the file.

"Obj_test.csv" is used to define the parameters and other attribues of the objective function tested. The benchmark objective functions used in this publication are provided in this file, but others can be added by adding an additional row to the file.
