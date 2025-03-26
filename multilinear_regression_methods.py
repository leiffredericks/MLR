import numpy as np
import scipy
import scipy.signal as sig
from typing import List, Union
import matplotlib.pyplot as plt
import warnings

# Methods packages
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression




def MLR_set(X, y, method='OLS', detrend=True, standardize=True,  weights=None, calibrate=True, calibration_X = None, calibration_y = None, 
            fit_intercept=False, EN_selection='random', ridge_solver='svd', l1_ratio=0.5, alpha=1, n_PLS_components=5, return_dynorm_dxnorm=False, random_seed=42):
    ############################################################################################################################
    ## Pre-processing the X matrix and y vector
    ############################################################################################################################

    # If provide one calibration, make sure both are provided
    if (calibration_X is None) ^ (calibration_y is None):
        raise Exception("You either need both a calibration X and y, or neither")

    # Remove any slope and mean from the data
    if detrend:
        X   = sig.detrend(X,axis=0)
        y   = sig.detrend(y,axis=0)

    # Need to save original X and y to be able to rescale later
    # Do after detrend since that is relevant to how variance is treated, but before standardize (facepalm)
    X_orig  = X
    y_orig  = y
    
    # Standardize both X and y data (even though it doesn't matter for fitting in y, just make everything units of sigma)
    if standardize:
        X   = (X - X.mean(axis=0)) / X.std(axis=0)
        y   = (y - y.mean(axis=0)) / y.std(axis=0)
        
    # If weights for X are included (predictor weights), weight the X variable
    if weights is not None:
        X   = X*weights    

    ############################################################################################################################
    ## Methods implementations
    ############################################################################################################################
    
    if method   == 'OLS':
        dy_dX   = np.linalg.lstsq(X, y)[0]

    elif method == 'MCA':
        X_i     = X.T # (space x time)
        Y       = y.reshape(-1,1) #(time x 1)
        m       = len(Y)
        # C_sz = <s @ z.T> where <> is time mean
        L_i     = 1/m * X_i @ Y  # (space x 1)
        # |L_i|, since SVD U is orthonormal so column 1 must have magnitude 1
        L_mag   = np.linalg.norm(L_i)
        # l_1 is equivalent to p_1 in this case
        l_1     = L_i/L_mag 
        dy_dX   = l_1.T[0]

    elif method =='RIDGE':
        clf     = Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=random_seed, solver=ridge_solver)
        clf.fit(X, y)
        dy_dX   = clf.coef_

    elif method == 'EN_RIDGE':
        clf     = ElasticNet(alpha=alpha, l1_ratio=0, fit_intercept=fit_intercept, random_state=random_seed,selection=EN_selection)
        clf.fit(X, y)
        dy_dX   = clf.coef_

    elif method == 'LASSO':
        clf     = ElasticNet(alpha=alpha, l1_ratio=1, fit_intercept=fit_intercept, random_state=random_seed, selection=EN_selection)
        clf.fit(X, y)
        dy_dX   = clf.coef_

    elif method == 'EN':
        clf     = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, random_state=random_seed, selection=EN_selection)
        clf.fit(X, y)
        dy_dX   = clf.coef_

    elif method == 'PLS':
        pls_opt = PLSRegression(n_components=n_PLS_components)
        pls_opt.fit(X, y)
        dy_dX   = pls_opt.coef_[0]
        
    else:
        supported_methods = {"OLS", "MCA", "RIDGE", "EN_RIDGE", "LASSO", "EN", "PLS"}
        raise ValueError(f"Unsupported method: {method}. Supported methods are {supported_methods}")

    ############################################################################################################################
    ## Post-processing partial derivatives / coefficients
    ############################################################################################################################

    # First incorporate weights, since the raw solution expects weighted X
    if weights is not None:
        dy_dX           = dy_dX*weights 

    # Second rescale the solution by the training variances to return to nominal units
    if standardize:
        # Save the sigma-unit solution before rescaling 
        dynorm_dXnorm   = dy_dX

        # If we are given a different unit variance to target
        if calibration_X is not None and calibration_y is not None:
            dy_dX       = dy_dX * np.std(calibration_y,axis=0) / np.std(calibration_X,axis=0)

        # Otherwise rescale to training variance to retrieve original units
        else:
            dy_dX       = dy_dX * np.std(y_orig,axis=0) / np.std(X_orig,axis=0)        

    # Third calibrate to recreate the correct variance 
    if calibrate:
        # If we are given a different variance to target
        if calibration_X is not None and calibration_y is not None:
            dy_dX       = dy_dX * np.std(calibration_y) / np.std(dy_dX@calibration_X.T)

        # Otherwise, recreate correct training variance (analogous to orthogonal regression)
        else:
            dy_dX       = dy_dX * np.std(y_orig) / np.std(dy_dX@X_orig.T)
    
    if return_dynorm_dxnorm:
        if not standardize:
            raise Exception("You did not specify standardizing the data, so there is no dynorm_dXnorm to return")
        return dy_dX, dynorm_dXnorm
    else:
        return dy_dX
    
def MLR_CV(X, y, method='OLS', detrend=True, standardize=True,  weights=None, calibrate=True, calibration_X = None, calibration_y = None, 
            cross_validation='resample_split', folds=5, n_resamples=10, resample_train_fraction=0.8, bounds=None, x0=None, 
            loss_func=None, tol=None, solver='Nelder-Mead', return_xVals=False, max_PLS_components=25, plot_PLS=False,
            fit_intercept=False, EN_selection='random', ridge_solver='svd', l1_ratio=0.5, alpha=1, n_PLS_components=5, return_dynorm_dxnorm=False, random_seed=42):
    
    # If provide one calibration, make sure both are provided
    if (calibration_X is None) ^ (calibration_y is None):
        raise Exception("You either need both a calibration X and y, or neither")

    # Initialize random number generator outside main cycle loop
    rng             = np.random.default_rng(seed=random_seed)

    if loss_func is None:
        # Default is RMSE
        def loss_func(dy_dX,X,y): return np.sqrt(np.mean((y-dy_dX@X.T)**2)) 

    # Define this outside the loop, after detrending
    def hyper_RIDGE(x, X_train, y_train, X_test, y_test):
        # Loss function vs alpha is smoother in log space
        alpha_here                  = 10**x[0]
        method_here                 = 'RIDGE'
        return_dynorm_dxnorm_here   = False

        dy_dX_here  =  MLR_set(X_train, y_train, method=method_here, alpha=alpha_here, return_dynorm_dxnorm=return_dynorm_dxnorm_here,
                    detrend=detrend,standardize=standardize,weights=weights,calibrate=calibrate,calibration_X=calibration_X,calibration_y=calibration_y, 
                    fit_intercept=fit_intercept,EN_selection=EN_selection,ridge_solver=ridge_solver,l1_ratio=l1_ratio,n_PLS_components=n_PLS_components,random_seed=random_seed)

        loss        = loss_func(dy_dX_here,X_test,y_test)

        return loss
    
    # Define this outside the loop, after detrending
    def hyper_EN_RIDGE(x, X_train, y_train, X_test, y_test):
        # Loss function vs alpha is smoother in log space
        alpha_here                  = 10**x[0] 
        method_here                 = 'EN_RIDGE'
        return_dynorm_dxnorm_here   = False

        dy_dX_here  =  MLR_set(X_train, y_train, method=method_here, alpha=alpha_here, return_dynorm_dxnorm=return_dynorm_dxnorm_here,standardize=standardize,
                        detrend=detrend,weights=weights,calibrate=calibrate,calibration_X=calibration_X,calibration_y=calibration_y, 
                        fit_intercept=fit_intercept,EN_selection=EN_selection,ridge_solver=ridge_solver,l1_ratio=l1_ratio,n_PLS_components=n_PLS_components,random_seed=random_seed)

        loss        = loss_func(dy_dX_here,X_test,y_test)

        return loss
    
    # Define this outside the loop, after detrending
    def hyper_LASSO(x, X_train, y_train, X_test, y_test):
        # Loss function vs alpha is smoother in log space
        alpha_here                  = 10**x[0] 
        method_here                 = 'LASSO'
        return_dynorm_dxnorm_here   = False

        dy_dX_here  =  MLR_set(X_train, y_train, method=method_here, alpha=alpha_here, return_dynorm_dxnorm=return_dynorm_dxnorm_here,standardize=standardize,
                        detrend=detrend,weights=weights,calibrate=calibrate,calibration_X=calibration_X,calibration_y=calibration_y, 
                        fit_intercept=fit_intercept,EN_selection=EN_selection,ridge_solver=ridge_solver,l1_ratio=l1_ratio,n_PLS_components=n_PLS_components,random_seed=random_seed)

        loss        = loss_func(dy_dX_here,X_test,y_test)

        return loss
    
    # Define this outside the loop, after detrending
    def hyper_EN(x, X_train, y_train, X_test, y_test):
        # Loss function vs alpha is smoother in log space
        alpha_here                  = 10**x[0] 
        l1_here                     = x[1]
        method_here                 = 'EN'
        return_dynorm_dxnorm_here   = False

        dy_dX_here  =  MLR_set(X_train, y_train, method=method_here, alpha=alpha_here,l1_ratio=l1_here, return_dynorm_dxnorm=return_dynorm_dxnorm_here,
                        standardize=standardize,detrend=detrend,weights=weights,calibrate=calibrate,calibration_X=calibration_X,calibration_y=calibration_y, 
                        fit_intercept=fit_intercept,EN_selection=EN_selection,ridge_solver=ridge_solver,n_PLS_components=n_PLS_components,random_seed=random_seed)

        loss        = loss_func(dy_dX_here,X_test,y_test)

        return loss

    # We will either cycle through the number of resamples requested or the number of folds in k-fold
    if cross_validation     == 'resample_split':
        n_cycles            = n_resamples
    elif cross_validation   == 'k-fold':
        n_cycles            = folds
        indices             = np.arange(len(y))
        # Shuffle indices randomly
        rng.shuffle(indices)
        # Split into k groups
        k_fold_groups   = np.array_split(indices, folds)
    else:
        supported_cross_validations = {'resample_split','k-fold'}
        raise ValueError(f"Unsupported cross-validation: {cross_validation}. Supported are {supported_cross_validations}")
    
    # Save the solution vector from each cross validation split
    xVals   = np.zeros((n_cycles,X.shape[1]))
    # Save the sigma units solution vector from each cross validation split
    xVals_norm   = np.zeros((n_cycles,X.shape[1]))
    # Save an array of hyperparams
    # Define a list that can hold floats or NumPy arrays
    hyper_params: List[Union[int, float, np.ndarray]] = []

    # Cycle through train/test splits 
    for i in np.arange(n_cycles):

        # How we select our mini train and mini test if we are randomly resampling
        if cross_validation == 'resample_split':
            # do a mini cross validation on random n:100-n train:test split
            choice          = rng.choice(range(len(y)), size=(np.int32(np.round(resample_train_fraction*len(y))),), replace=False) # Pick n% of the indices
            train           = np.zeros(len(y), dtype=bool)  # Initialize trianing indices
            train[choice]   = True      # Assign the n% to the trianing indices
            test            = ~train    # The rest are testing indices
            
            mini_train_X    = X[train]
            mini_train_y    = y[train]
            mini_test_X     = X[test]
            mini_test_y     = y[test]

        # How we select our mini train and mini test if we are doing k-fold
        elif cross_validation       == 'k-fold':
            test                    = np.zeros(len(y), dtype=bool) # Initialize testing indices (fold)
            test[k_fold_groups[i]]  = True     # Assign the ith group as the held-out test
            train                   = ~test    # The rest are training indices

            mini_train_X    = X[train]
            mini_train_y    = y[train]
            mini_test_X     = X[test]
            mini_test_y     = y[test]

        else:
            supported_cross_validations = {'resample_split','k-fold'}
            raise ValueError(f"Unsupported cross-validation: {cross_validation}. Supported are {supported_cross_validations}")
            
        if method   == 'PLS':
            scores  = np.zeros(max_PLS_components)
            for j in np.arange(max_PLS_components):
                # Top row of options matter, rest carry through from global options
                dy_dX_here  = MLR_set(mini_train_X, mini_train_y, method='PLS', n_PLS_components=np.int32(j+1), return_dynorm_dxnorm=False,
                                    detrend=detrend, standardize=standardize,  weights=weights, calibrate=calibrate, 
                                    calibration_X=calibration_X, calibration_y=calibration_y, fit_intercept=fit_intercept, 
                                    EN_selection=EN_selection, ridge_solver=ridge_solver, l1_ratio=l1_ratio, alpha=alpha, random_seed=random_seed)
                scores[j]   = loss_func(dy_dX_here, mini_test_X, mini_test_y)

            # Pull out best number of components to keep
            num_components  = np.arange(max_PLS_components)+1
            n_best          = np.int32(num_components[np.argmin(scores)])
            hyper_params.append(n_best)

            # Apply to full training set. Top row of options matter, rest carry through from global options
            if return_dynorm_dxnorm:
                xVals[i], xVals_norm[i] = MLR_set(X, y, method='PLS', n_PLS_components=n_best, return_dynorm_dxnorm=return_dynorm_dxnorm,
                                        detrend=detrend, standardize=standardize,  weights=weights, calibrate=calibrate, 
                                        calibration_X=calibration_X, calibration_y=calibration_y, fit_intercept=fit_intercept, 
                                        EN_selection=EN_selection, ridge_solver=ridge_solver, l1_ratio=l1_ratio, alpha=alpha, random_seed=random_seed)
            else:
                xVals[i]                = MLR_set(X, y, method='PLS', n_PLS_components=n_best, return_dynorm_dxnorm=return_dynorm_dxnorm,
                                        detrend=detrend, standardize=standardize,  weights=weights, calibrate=calibrate, 
                                        calibration_X=calibration_X, calibration_y=calibration_y, fit_intercept=fit_intercept, 
                                        EN_selection=EN_selection, ridge_solver=ridge_solver, l1_ratio=l1_ratio, alpha=alpha, random_seed=random_seed)

            if plot_PLS:
                plt.plot(num_components,scores)
                plt.title('Cross-validation from number of PLS components kept')
                plt.xlabel('# of PLS components')
                plt.ylabel('Loss')
                plt.show()
            continue
            
        elif method   == 'OLS':
            warnings.warn("OLS has no parameters to set via cross-validation. Returning the OLS solution.", UserWarning)
            return MLR_set(X, y, method='OLS', n_PLS_components=n_PLS_components, return_dynorm_dxnorm=return_dynorm_dxnorm,
                                    detrend=detrend, standardize=standardize,  weights=weights, calibrate=calibrate, 
                                    calibration_X=calibration_X, calibration_y=calibration_y, fit_intercept=fit_intercept, 
                                    EN_selection=EN_selection, ridge_solver=ridge_solver, l1_ratio=l1_ratio, alpha=alpha, random_seed=random_seed)

        elif method == 'MCA':
            warnings.warn("MCA has no parameters to set via cross-validation. Returning the MCA solution.", UserWarning)
            return MLR_set(X, y, method='MCA', n_PLS_components=n_PLS_components, return_dynorm_dxnorm=return_dynorm_dxnorm,
                                    detrend=detrend, standardize=standardize,  weights=weights, calibrate=calibrate, 
                                    calibration_X=calibration_X, calibration_y=calibration_y, fit_intercept=fit_intercept, 
                                    EN_selection=EN_selection, ridge_solver=ridge_solver, l1_ratio=l1_ratio, alpha=alpha, random_seed=random_seed)


        elif method =='RIDGE':
            if x0 is None:
                x0          = [10]
            if bounds is None:
                bounds      = [(1e-7,1e6)] 

            # log 10 scale in hyper_RIDGE, so need to adjust 
            x0_here         = np.log10(x0)
            bounds_here     = np.log10(bounds)

            result_here     = scipy.optimize.minimize(hyper_RIDGE, x0_here, args=(mini_train_X,mini_train_y,mini_test_X,mini_test_y), method=solver, bounds=bounds_here, tol=tol)

            # Loss function vs alpha is smoother in log space, convert back to alpha
            alpha_here      = 10**result_here.x[0]
            hyper_params.append(alpha_here)

            # Apply to full training set. Top row of options matter, rest carry through from global options
            if return_dynorm_dxnorm:
                xVals[i], xVals_norm[i] = MLR_set(X, y, method='RIDGE', alpha=alpha_here, return_dynorm_dxnorm=return_dynorm_dxnorm,
                                        detrend=detrend,standardize=standardize,weights=weights,calibrate=calibrate,calibration_X=calibration_X,calibration_y=calibration_y, 
                                        fit_intercept=fit_intercept,EN_selection=EN_selection,ridge_solver=ridge_solver,l1_ratio=l1_ratio,n_PLS_components=n_PLS_components,
                                        random_seed=random_seed)
            else:
                xVals[i]                = MLR_set(X, y, method='RIDGE', alpha=alpha_here, return_dynorm_dxnorm=return_dynorm_dxnorm,
                                        detrend=detrend,standardize=standardize,weights=weights,calibrate=calibrate,calibration_X=calibration_X,calibration_y=calibration_y, 
                                        fit_intercept=fit_intercept,EN_selection=EN_selection,ridge_solver=ridge_solver,l1_ratio=l1_ratio,n_PLS_components=n_PLS_components,
                                        random_seed=random_seed)

        elif method == 'EN_RIDGE':
            if x0 is None:
                x0          = [10]
            if bounds is None:
                bounds      = [(1e-7,1e3)] 
            
            # log 10 scale in hyper_EN_RIDGE, so need to adjust 
            x0_here         = np.log10(x0)
            bounds_here     = np.log10(bounds)

            result_here     = scipy.optimize.minimize(hyper_EN_RIDGE, x0_here, args=(mini_train_X,mini_train_y,mini_test_X,mini_test_y), method=solver, bounds=bounds_here, tol=tol)

            # Loss function vs alpha is smoother in log space, convert back to alpha
            alpha_here      = 10**result_here.x[0]
            hyper_params.append(alpha_here)

            # Apply to full training set. Top row of options matter, rest carry through from global options
            if return_dynorm_dxnorm:
                xVals[i], xVals_norm[i] = MLR_set(X, y, method='EN_RIDGE', alpha=alpha_here, return_dynorm_dxnorm=return_dynorm_dxnorm,
                                        detrend=detrend,standardize=standardize,weights=weights,calibrate=calibrate,calibration_X=calibration_X,calibration_y=calibration_y, 
                                        fit_intercept=fit_intercept,EN_selection=EN_selection,ridge_solver=ridge_solver,l1_ratio=l1_ratio,n_PLS_components=n_PLS_components,
                                        random_seed=random_seed)
            else:
                xVals[i]                = MLR_set(X, y, method='EN_RIDGE', alpha=alpha_here, return_dynorm_dxnorm=return_dynorm_dxnorm,
                                        detrend=detrend,standardize=standardize,weights=weights,calibrate=calibrate,calibration_X=calibration_X,calibration_y=calibration_y, 
                                        fit_intercept=fit_intercept,EN_selection=EN_selection,ridge_solver=ridge_solver,l1_ratio=l1_ratio,n_PLS_components=n_PLS_components,
                                        random_seed=random_seed)

        elif method == 'LASSO':
            if x0 is None:
                x0          = [0.1]
            if bounds is None:
                bounds      = [(1e-9,1e2)] 

            # log 10 scale in hyper_LASSO, so need to adjust 
            x0_here         = np.log10(x0)
            bounds_here     = np.log10(bounds)

            result_here     = scipy.optimize.minimize(hyper_LASSO, x0_here, args=(mini_train_X,mini_train_y,mini_test_X,mini_test_y), method=solver, bounds=bounds_here, tol=tol)

            # Loss function vs alpha is smoother in log space, convert back to alpha
            alpha_here      = 10**result_here.x[0]
            hyper_params.append(alpha_here)

            # Apply to full training set. Top row of options matter, rest carry through from global options
            if return_dynorm_dxnorm:
                xVals[i], xVals_norm[i] = MLR_set(X, y, method='LASSO', alpha=alpha_here, return_dynorm_dxnorm=return_dynorm_dxnorm,
                                            detrend=detrend,standardize=standardize,weights=weights,calibrate=calibrate,calibration_X=calibration_X,calibration_y=calibration_y, 
                                            fit_intercept=fit_intercept,EN_selection=EN_selection,ridge_solver=ridge_solver,l1_ratio=l1_ratio,n_PLS_components=n_PLS_components,
                                            random_seed=random_seed)
            else:
                xVals[i]                = MLR_set(X, y, method='LASSO', alpha=alpha_here, return_dynorm_dxnorm=return_dynorm_dxnorm,
                                            detrend=detrend,standardize=standardize,weights=weights,calibrate=calibrate,calibration_X=calibration_X,calibration_y=calibration_y, 
                                            fit_intercept=fit_intercept,EN_selection=EN_selection,ridge_solver=ridge_solver,l1_ratio=l1_ratio,n_PLS_components=n_PLS_components,
                                            random_seed=random_seed)

        elif method == 'EN':
            if x0 is None:
                x0          = [0.1,0.5]
            if bounds is None:
                bounds      = [(1e-9,1e3), (1e-9, 0.99)] 

            # log 10 scale in hyper_EN for alpha, so need to adjust 
            x0_here         = [np.log10(x0[0]),x0[1]]
            bounds_here     = [np.log10(bounds[0]),bounds[1]]

            result_here     = scipy.optimize.minimize(hyper_EN, x0_here, args=(mini_train_X,mini_train_y,mini_test_X,mini_test_y), method=solver, bounds=bounds_here, tol=tol)

            # Loss function vs alpha is smoother in log space, convert back to alpha
            alpha_here      = 10**result_here.x[0]
            l1_here         = result_here.x[1]
            hyper_params.append(np.array([alpha_here,l1_here]))

            # Apply to full training set. Top row of options matter, rest carry through from global options
            if return_dynorm_dxnorm:
                xVals[i], xVals_norm[i] = MLR_set(X, y, method='EN', alpha=alpha_here, l1_ratio=l1_here, return_dynorm_dxnorm=return_dynorm_dxnorm,
                                        detrend=detrend,standardize=standardize,weights=weights,calibrate=calibrate,calibration_X=calibration_X,calibration_y=calibration_y, 
                                        fit_intercept=fit_intercept,EN_selection=EN_selection,ridge_solver=ridge_solver,n_PLS_components=n_PLS_components,
                                        random_seed=random_seed)
            else:
                xVals[i]                = MLR_set(X, y, method='EN', alpha=alpha_here, l1_ratio=l1_here, return_dynorm_dxnorm=return_dynorm_dxnorm,
                                        detrend=detrend,standardize=standardize,weights=weights,calibrate=calibrate,calibration_X=calibration_X,calibration_y=calibration_y, 
                                        fit_intercept=fit_intercept,EN_selection=EN_selection,ridge_solver=ridge_solver,n_PLS_components=n_PLS_components,
                                        random_seed=random_seed)
    
        else:
            supported_methods = {"OLS", "MCA", "RIDGE", "EN_RIDGE", "LASSO", "EN", "PLS"}
            raise ValueError(f"Unsupported method: {method}. Supported methods are {supported_methods}")
    
    if return_xVals:
        if return_dynorm_dxnorm:
            return xVals, xVals_norm, hyper_params
        else:
            return xVals, hyper_params
    else:
        if return_dynorm_dxnorm:
            return np.nanmean(xVals,axis=0), np.nanmean(xVals_norm,axis=0), hyper_params
        else:
            return np.nanmean(xVals,axis=0), hyper_params

