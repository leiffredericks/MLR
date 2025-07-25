# MLR
These functions allow one to apply multilinear regression from a 1-dimensional series y onto a 2-dimensional field series X. Supported methods are:
- Ordinary Least Squares
- Maximum Covariance Analysis
- Ridge Regression
- LASSO Regression
- Elastic Net Regression
- Partial Least Squares

Included are many options for how to process the data and return the solution vector of regression coefficients. Review options set to "True" by default before using to make sure the function is doing what is expected.

# MLR_set
Performs multilinear regression (MLR) using one of the supported methods on the training data with the hyperparameters set by the user. Makes the basic assumption that the system can be described linearly as y=X@dy_dX + ε. Returns the vector of predictor coefficients dy_dX.

# MLR_CV
Performs cross-validation to set the parameters required by the methods supported by MLR_set. Makes the basic assumption that the system can be described linearly as y=X@dy_dX + ε. Returns the optimized solution(s) and the parameter(s) used to produce those solution(s). 
