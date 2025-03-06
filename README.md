# MLR

# MLR_set
Performs multilinear regression (MLR) using one of the supported methods on the training data with the hyperparameters set by the user. Makes the basic assumption that the system can be described linearly as y=X@dy_dX + ε. Returns the vector of predictor coefficients dy_dX.

# MLR_CV
Performs cross-validation to set the hyperparameters required by the methods supported by MLR_set. Makes the basic assumption that the system can be described linearly as y=X@dy_dX + ε. Returns the optimized solution(s) and the hyperparameter(s) used to produce those solution(s). 
