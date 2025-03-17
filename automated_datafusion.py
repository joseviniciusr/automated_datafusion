"""
**Python library to easily automate data fusion models (low-, mid-, high-levels)**
 
**Description**:

Additionally to modeling (by PLS, RF or SVM), the individual modeling function independently extracts the Latent Variable Scores (-based on PLS transform).

The data fusion functions are mainly powered by dictionaries where the keys represent all employed sensors (m). Then, all possible combinations ranging from 2 to m are modeled along with several performance metrics are extracted.

**Author**: Msc. Jose Vinicius Ribeiro

**Applied Nuclear Physics Laboratory**, State University of Londrina, Brazil

**Contact**: ribeirojosevinicius@gmail.com
"""

import pandas as pd
import numpy as np

def modelo_individual_otimizado(Xcal, ycal, target, model='pls', Xpred=None, ypred=None, 
                                maxLV=None, kern=None, random_seed=None, LVscores=None):
    """
    ## Individual modeling (calibration and prediction) via pls, rf or svm
     **Latent variables are extractedas well**

    **Input parameters**:
    - **Xcal**: pd.DataFrame with the X calibration data
    - **Xpred**: pd.DataFrame with the X prediction data (Optional)
    - **Ycal**: pd.DataFrame with the y calibration data. The columns should be named according to the "target" input
    - **Ypred**: pd.DataFrame with the y calibration data. The columns should be named according to the "target" input (Optional)
    - **target**: string representing the target variable (according to the column names of Ycal/Ypred)
    - **model**: the model to be used, chosen among 'pls', 'rf', 'svm'
    - **maxLV**: If model='pls', maximum number of LVs to be used
    - **kern**: If model='svm', the kernel must be chosen among 'linear', 'rbf', 'sigmoid' 
    - **random_seed**: Int. If model='rf', the random seed should be chosen

    **Return**:
    - If LVscores = True:

         (**df_results**, **calres**, **lv_scorescal**, **predres**, **lv_scorespred**)         
    - Else:

         (**df_results**, **calres**, **predres**)

    **Where**:
    - **df_results**: DataFrame with performance metrics
    - **calres**: DataFrame with predictions in calibration (with 'Ref' column of real values).
    - **lv_scorescal**: DataFrame with the LV scores in the calibration.
    - **predres**: DataFrame with the predictions in the prediction (with 'Ref' column of actual values, if available).
    - **lv_scorespred**: DataFrame with the LV scores in the prediction.     
    """

    # List to store metric results
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import iqr

    results = []
    ycal=ycal[target]
    ypred=ypred[target]

    # DataFrames to store the predicted calibration and prediction values ​​(if provided by Xpred)
    calres = pd.DataFrame(index=range(len(ycal)))
    predres = pd.DataFrame(index=range(len(ypred))) if Xpred is not None and ypred is not None else None

    if model == 'pls':
        if maxLV is None:
            print("For the 'pls' model, enter the desired number of latent variables in the LV argument")
            return None, None, None, None, None
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.model_selection import cross_val_predict
        # Loop for each number of latent variables (from 1 to n_max_comp)
        kern = None
        random_seed = None
        for n_comp in range(1, maxLV + 1):
            # Define the PLS model with the current number of latent variables
            pls = PLSRegression(n_components=n_comp, scale=False)
            
            # Fit the model to the calibration data
            pls.fit(Xcal, ycal)
            y_cal = pls.predict(Xcal).flatten()  # Predicted values ​​for calibration (flattens to 1D)

            # Adds column of predicted calibration values ​​for current number of LVs
            calres[f'LV_{n_comp}'] = y_cal
            
            # Cross-validation
            y_cv = cross_val_predict(pls, Xcal, ycal, cv=10)
            
            # Calculates calibration metrics
            R2_cal = r2_score(ycal, y_cal)
            r2_cal = (np.corrcoef(ycal, y_cal)[0, 1]) ** 2
            rmse_cal = np.sqrt(mean_squared_error(ycal, y_cal))

            # Calculates cross-validation metrics
            R2_cv = r2_score(ycal, y_cv)
            r2_cv = (np.corrcoef(ycal, y_cv)[0, 1]) ** 2
            rmsecv = np.sqrt(mean_squared_error(ycal, y_cv))
            bias_cv = sum(ycal - y_cv)/ycal.shape[0]
            SDV_cv = (ycal - y_cv) - bias_cv
            SDV_cv = SDV_cv*SDV_cv
            SDV_cv = np.sqrt(sum(SDV_cv)/(ycal.shape[0] - 1))
            tbias_cv = abs(bias_cv)*(np.sqrt(ycal.shape[0])/SDV_cv)
            rpd_cv = ycal.std() / rmsecv
            rpiq_cv = iqr(ycal, rng=(25, 75)) / rmsecv

            # Checks if Xpred and ypred were provided to calculate the prediction metrics
            if Xpred is not None and ypred is not None:
                # Makes prediction for the prediction dataset
                y_pred = pls.predict(Xpred).flatten()
                
                # Stores predicted values ​​in predres
                predres[f'LV_{n_comp}'] = y_pred
                
                # Calculates prediction metrics
                R2_pred = r2_score(ypred, y_pred)
                r2_pred = (np.corrcoef(ypred, y_pred)[0, 1]) ** 2
                rmsep = np.sqrt(mean_squared_error(ypred, y_pred))
                bias_pred = sum(ypred - y_pred)/ypred.shape[0]
                SDV_pred = (ypred - y_pred) - bias_pred
                SDV_pred = SDV_pred*SDV_pred
                SDV_pred = np.sqrt(sum(SDV_pred)/(ypred.shape[0] - 1))
                tbias_pred = abs(bias_pred)*(np.sqrt(ypred.shape[0])/SDV_pred)
                rpd_pred = ypred.std() / rmsep
                rpiq_pred = iqr(ypred, rng=(25, 75)) / rmsep
            else:
                # Sets the prediction metric values ​​to None if Xpred or ypred are not provided
                R2_pred = r2_pred = rmsep = rpd_pred = rpiq_pred = None

            results.append({
                'LVs number': n_comp,
                'R2 Cal': R2_cal,
                'r2 Cal': r2_cal,
                'RMSEC': rmse_cal,
                'R2 CV': R2_cv,
                'r2 CV': r2_cv,
                'RMSECV': rmsecv,
                'Bias CV': bias_cv,
                'tbias CV': tbias_cv,
                'RPD CV': rpd_cv,
                'RPIQ CV': rpiq_cv,
                'R2 Pred': R2_pred,
                'r2 Pred': r2_pred,
                'RMSEP': rmsep,
                'Bias Pred': bias_pred,
                'tbias Pred': tbias_pred,
                'RPD Pred': rpd_pred,
                'RPIQ Pred': rpiq_pred
            })    
            
    elif model == 'rf':
        if random_seed is None:
            print("For the 'rf' model, enter the desired seed number in random_seed")
            return None, None, None, None, None
        kern = None
        # Fit a Random Forest model with default hyperparameters
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.cross_decomposition import PLSRegression
        rf = RandomForestRegressor(criterion='squared_error',
                                   random_state=random_seed)
        
        #Calibration
        rf.fit(Xcal, ycal)
        ycal_res = rf.predict(Xcal)
        calres['RF'] = ycal_res

        # Calculates calibration metrics
        R2_cal = r2_score(ycal, ycal_res)
        r2_cal = (np.corrcoef(ycal, ycal_res)[0, 1]) ** 2
        rmse_cal = np.sqrt(mean_squared_error(ycal, ycal_res))

        # Checks if Xpred and ypred were provided to calculate the prediction metrics
        if Xpred is not None and ypred is not None:
            ypred_res = rf.predict(Xpred)
            predres['RF'] = ypred_res

            # Calculates prediction metrics
            R2_pred = r2_score(ypred, ypred_res)
            r2_pred = (np.corrcoef(ypred, ypred_res)[0, 1]) ** 2
            rmse_pred = np.sqrt(mean_squared_error(ypred, ypred_res))
            bias_pred = sum(ypred - ypred_res)/ypred.shape[0]
            SDV_pred = (ypred - ypred_res) - bias_pred
            SDV_pred = SDV_pred*SDV_pred
            SDV_pred = np.sqrt(sum(SDV_pred)/(ypred.shape[0] - 1))
            tbias_pred = abs(bias_pred)*(np.sqrt(ypred.shape[0])/SDV_pred)
            rpd_pred = ypred.std() / rmse_pred
            rpiq_pred = iqr(ypred, rng=(25, 75)) / rmse_pred
        else: 
            R2_pred = r2_pred = rmse_pred = rpd_pred = rpiq_pred = None
    
        # Stores the results in a dictionary
        results.append({
            'Model': 'RF',
            'R2 Cal': R2_cal,
            'r2 Cal': r2_cal,
            'RMSEC': rmse_cal,
            'R2 Pred': R2_pred,
            'r2 Pred': r2_pred,
            'RMSEP': rmse_pred,
            'Bias Pred': bias_pred,
            'tbias Pred': tbias_pred,
            'RPD Pred': rpd_pred,
            'RPIQ Pred': rpiq_pred
        })
        
    elif model == 'svm':
        if kern is None:
            print("For the 'svm' model choose a kernel among 'linear', 'poly', 'rbf', 'sigmoid' using the kern argument")
            return None, None, None, None, None
        maxLV = None 
        random_seed = None   
        from sklearn.svm import SVR
        from sklearn.cross_decomposition import PLSRegression
        svm = SVR(kernel=kern)
        svm.fit(Xcal, ycal)

        #Calibration
        ycal_res = svm.predict(Xcal)
        calres['SVM'] = ycal_res

        # Calculates calibration metrics
        R2_cal = r2_score(ycal, ycal_res)
        r2_cal = (np.corrcoef(ycal, ycal_res)[0, 1]) ** 2
        rmse_cal = np.sqrt(mean_squared_error(ycal, ycal_res))

        # Checks if Xpred and ypred were provided to calculate the prediction metrics
        if Xpred is not None and ypred is not None:
            ypred_res = svm.predict(Xpred)
            predres['SVM'] = ypred_res

            # Calculates prediction metrics
            R2_pred = r2_score(ypred, ypred_res)
            r2_pred = (np.corrcoef(ypred, ypred_res)[0, 1]) ** 2
            rmse_pred = np.sqrt(mean_squared_error(ypred, ypred_res))
            bias_pred = sum(ypred - ypred_res)/ypred.shape[0]
            SDV_pred = (ypred - ypred_res) - bias_pred
            SDV_pred = SDV_pred*SDV_pred
            SDV_pred = np.sqrt(sum(SDV_pred)/(ypred.shape[0] - 1))
            tbias_pred = abs(bias_pred)*(np.sqrt(ypred.shape[0])/SDV_pred)
            rpd_pred = ypred.std() / rmse_pred
            rpiq_pred = iqr(ypred, rng=(25, 75)) / rmse_pred
        else: 
            R2_pred = r2_pred = rmse_pred = rpd_pred = rpiq_pred = None
                
        # Stores the results in a dictionary
        results.append({
            'Model': 'SVM',
            'R2 Cal': R2_cal,
            'r2 Cal': r2_cal,
            'RMSEC': rmse_cal,
            'R2 Pred': R2_pred,
            'r2 Pred': r2_pred,
            'RMSEP': rmse_pred,
            'Bias Pred': bias_pred,
            'tbias Pred': tbias_pred,
            'RPD Pred': rpd_pred,
            'RPIQ Pred': rpiq_pred
        })

    # Converts the list of results to a DataFrame
    df_results = pd.DataFrame(results)
    calres.insert(0, 'Ref', ycal)
    if predres is not None:
        predres.insert(0, 'Ref', ypred)

    # Independent extraction of LV scores, if requested and if maxLV is provided
    if LVscores is not None and maxLV is not None:
        from sklearn.cross_decomposition import PLSRegression
        
        pls_scores = PLSRegression(n_components=maxLV, scale=False)
        pls_scores.fit(Xcal, ycal)
        lv_scorescal = pd.DataFrame(pls_scores.transform(Xcal),
                                    columns=[f'LV_{i+1}' for i in range(maxLV)])
        if Xpred is not None and ypred is not None:
            lv_scorespred = pd.DataFrame(pls_scores.transform(Xpred),
                                         columns=[f'LV_{i+1}' for i in range(maxLV)])
        else:
            lv_scorespred = None
    else:
        lv_scorescal = None
        lv_scorespred = None

    # Conditional return: if LVscores is True, also return the LV scores dataFrames
    # Otherwise, return only the results and the calibration and prediction dataFrames.
    if LVscores:
        return df_results, calres, lv_scorescal, predres, lv_scorespred
    else:
        return df_results, calres, predres
    
# function to generate high-level fusion models with all possible combinations of predictors
def high_level_fusion_automatizado(calres, predres, ycal, ypred, target):
    """
    ## High-level data fusion. 
    **Generates multiple linear regression models for all possible combinations of inputs.**

    Parâmetros:
    - **calres_dict**: dictionary of DataFrames with the calibration data.
    - **predres_dict**: dictionary of DataFrames with the prediction data.
    - **Ycal**: DataFrame containing the actual values ​​for calibration.
    - **Ypred**: DataFrame containing the actual values ​​for prediction.
    - **target**: string representing the target variable.

    **Retorna**:
    - results: dictionary containing the models, coefficients, predictions and metrics for each possible combination.
    """

    import itertools
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import iqr
    import numpy as np
    import pandas as pd
    results = {}

    # Number of columns desired per combination (from 2 to all)
    for i in range(2, len(calres) + 1):
        for combinacoes in itertools.combinations(calres.keys(), i):
            # itertools.combinations(calres_dict.keys(), r): Generates all possible combinations of size r using the keys of the dictionary
    
            # Create DataFrame with only the last columns of the chosen combinations
            high_level_cal = pd.DataFrame({key: calres[key].iloc[:, -1] for key in combinacoes})
            high_level_pred = pd.DataFrame({key: predres[key].iloc[:, -1] for key in combinacoes})

            high_level_cal.columns = high_level_cal.columns.astype(str)
            high_level_pred.columns = high_level_pred.columns.astype(str)

            # Set actual values ​​for calibration and prediction
            ytrain = ycal[target].values  # Convert to NumPy array to avoid warnings
            ytest = ypred[target].values

            # Create and train the multiple linear regression model
            model = LinearRegression()
            model.fit(high_level_cal, ytrain)

            # Predictions in calibration and testing
            ycal_res = model.predict(high_level_cal)
            ypred_res = model.predict(high_level_pred)

            # Calculation of metrics for calibration
            R2_cal = r2_score(ytrain, ycal_res)
            r2_cal = (np.corrcoef(ytrain, ycal_res)[0, 1]) ** 2
            rmse_cal = np.sqrt(mean_squared_error(ytrain, ycal_res))

            # Calculation of metrics for prediction
            R2_pred = r2_score(ytest, ypred_res)
            r2_pred = (np.corrcoef(ytest, ypred_res)[0, 1]) ** 2
            rmse_pred = np.sqrt(mean_squared_error(ytest, ypred_res))
            bias_pred = sum(ytest - ypred_res)/ytest.shape[0]
            SDV_pred = (ytest - ypred_res) - bias_pred
            SDV_pred = SDV_pred*SDV_pred
            SDV_pred = np.sqrt(sum(SDV_pred)/(ytest.shape[0] - 1))
            tbias_pred = abs(bias_pred)*(np.sqrt(ytest.shape[0])/SDV_pred)
            rpd = ytest.std() / rmse_pred
            rpiq = iqr(ytest, rng=(25, 75)) / rmse_pred

            # Create DataFrame with predictions
            predicoes = pd.DataFrame({
                'Results': np.concatenate((ycal_res, ypred_res)),  # Predicted values
                'Ref': np.concatenate((ytrain, ytest)),  # Combined real values
                'Set': ['cal'] * len(ytrain) + ['pred'] * len(ytest)  # Set indication
            })

            combinacoes_name = "_".join(combinacoes)
            # Joins the names of selected columns by combining with an underscore
                        
            # Store results in the results dictionary
            results[combinacoes_name] = {
                'model': model,
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'predictions': predicoes,  
                'metrics': {
                    'R2 Cal': R2_cal,
                    'r2 Cal': r2_cal,
                    'RMSEC': rmse_cal,
                    'R2 Pred': R2_pred,
                    'r2_pred': r2_pred,
                    'RMSEP': rmse_pred,
                    'Bias Pred': bias_pred,
                    'tbias Pred': tbias_pred,
                    'RPD Pred': rpd,
                    'RPIQ Pred': rpiq
                }
            }

    return results

def mid_level_fusion_automatizado(scorescal, scorespred, ycal, ypred, target):
    """
    ## Mid-level data fusion
    **Generates multiple linear regression models for all possible combinations of inputs.**

    **Parameters**:
    - **scorescal**: dictionary of DataFrames with the calibration scores data.
    - **scorespred**: dictionary of DataFrames with the prediction scores data.
    - **ycal**: DataFrame containing the actual values ​​for calibration.
    - **ypred**: DataFrame containing the actual values ​​for prediction.
    - **target**: string representing the target variable.

    **Retorna**: dictionary containing the models, coefficients, predictions and metrics for each possible combination.
    """
    import itertools
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import iqr
    import numpy as np
    import pandas as pd

    results = {}

    # Number of columns desired per combination (from 2 to all)
    for i in range(2, len(scorescal) + 1):
        for combinacoes in itertools.combinations(scorescal.keys(), i):
            mid_level_cal = pd.concat([scorescal[key] for key in combinacoes], axis=1)
            mid_level_pred = pd.concat([scorespred[key] for key in combinacoes], axis=1)

            # Convert column names to string to avoid error in PLSRegression
            mid_level_cal.columns = mid_level_cal.columns.astype(str)
            mid_level_pred.columns = mid_level_pred.columns.astype(str)

            # Set actual values ​​for calibration and prediction
            ytrain = ycal[target].values  
            ytest = ypred[target].values

            # Create and train the multiple linear regression model
            model = LinearRegression()
            model.fit(mid_level_cal, ytrain)

            # Predictions in calibration and testing
            ycal_res = model.predict(mid_level_cal)
            ypred_res = model.predict(mid_level_pred)

            # Calculation of metrics for calibration
            R2_cal = r2_score(ytrain, ycal_res)
            r2_cal = (np.corrcoef(ytrain, ycal_res)[0, 1]) ** 2
            rmse_cal = np.sqrt(mean_squared_error(ytrain, ycal_res))

            # Calculation of metrics for prediction
            R2_pred = r2_score(ytest, ypred_res)
            r2_pred = (np.corrcoef(ytest, ypred_res)[0, 1]) ** 2
            rmse_pred = np.sqrt(mean_squared_error(ytest, ypred_res))
            bias_pred = sum(ytest - ypred_res)/ytest.shape[0]
            SDV_pred = (ytest - ypred_res) - bias_pred
            SDV_pred = SDV_pred*SDV_pred
            SDV_pred = np.sqrt(sum(SDV_pred)/(ytest.shape[0] - 1))
            tbias_pred = abs(bias_pred)*(np.sqrt(ytest.shape[0])/SDV_pred)
            rpd = ytest.std() / rmse_pred
            rpiq = iqr(ytest, rng=(25, 75)) / rmse_pred

            # Create DataFrame with predictions
            predicoes = pd.DataFrame({
                'Results': np.concatenate((ycal_res, ypred_res)),  
                'Ref': np.concatenate((ytrain, ytest)), 
                'Set': ['cal'] * len(ytrain) + ['pred'] * len(ytest) 
            })

            combinacoes_name = "_".join(combinacoes)
            
            results[combinacoes_name] = {
                'model': model,
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'predictions': predicoes, 
                'metrics': {
                    'R2 Cal': R2_cal,
                    'r2 Cal': r2_cal,
                    'RMSEC': rmse_cal,
                    'R2 Pred': R2_pred,
                    'r2_pred': r2_pred,
                    'RMSEP': rmse_pred,
                    'Bias Pred': bias_pred,
                    'tbias Pred': tbias_pred,
                    'RPD Pred': rpd,
                    'RPIQ Pred': rpiq
                }
            }

    return results

def auto_scaling(input_data):
    """
    Auto scaling
    ----------
    Returns the preprocessed data, sd in the original space and mean in the original space
    """
    mean_original = np.mean(input_data, axis=0)
    sd_original = np.std(input_data, axis=0)
    X_sc = input_data / sd_original
    X_sc = X_sc - mean_original
    X_sc[np.isnan(X_sc)] = 0
    return pd.DataFrame(X_sc), sd_original, mean_original

def low_level_fusion_automatizado(espectroscal, espectrospred, ycal, ypred, target, scale=True, LV=None, model='pls', kern=None, random_seed=None):
    """
    ## Low-level data fusion
    **Generates models for all possible combinations of inputs (low level fusion) using PLS, Random Forest or SVM.**
    
    **Parâmetros**:
    - **espectroscal**: dictionary of DataFrames with the calibration spectra.
    - **espectrospred**: dictionary of DataFrames with the prediction spectra.
    - **ycal**: pd.DataFrame with the y calibration data. The columns should be named according to the "target" input
    - **ypred**: pd.DataFrame with the y calibration data. The columns should be named according to the "target" input (Optional)
    - **target**: string representing the target variable (according to the column names of Ycal/Ypred)
    - **scale**: Bol that says whether the data will be autoscaled before modeling or not
    - **model**: the model to be used, chosen among 'pls', 'rf', 'svm'
    - **LV**: If model='pls', maximum number of LVs to be used
    - **kern**: If model='svm', the kernel must be chosen among 'linear', 'rbf', 'sigmoid' 
    - **random_seed**: Int. If model='rf', the random seed should be chosen
    - **Note**: the pls model implements a 10-fold cross-validation
    
    **Return**: results: nested dictionary containing the models, predictions, metrics, all the important information
    """
    import itertools
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import iqr

    model = model.lower()
    if model not in ['pls', 'rf', 'svm']:
        print("The 'model' parameter must be 'pls', 'rf' or svm")

    results = {}

    # Iterates through all possible combinations (from 2 to all available sources)
    for i in range(2, len(espectroscal) + 1):
        for combinacoes in itertools.combinations(espectroscal.keys(), i):
            # Concatenate spectra for calibration and prediction
            low_level_cal = pd.concat([espectroscal[key] for key in combinacoes], axis=1)
            low_level_pred = pd.concat([espectrospred[key] for key in combinacoes], axis=1)

            # Ensures column names are strings
            low_level_cal.columns = low_level_cal.columns.astype(str)
            low_level_pred.columns = low_level_pred.columns.astype(str)
            
            if scale:
                low_level_cal, sdcal, meancal = auto_scaling(low_level_cal) # autoscaling the data
                low_level_pred = ( low_level_pred / sdcal ) - meancal
                low_level_pred[np.isnan(low_level_pred)] = 0
            else:
                low_level_cal = low_level_cal
                low_level_pred = low_level_pred
        
            combinacoes_name = "_".join(combinacoes)
            results[combinacoes_name] = {}
            ytrain = ycal[target].values
            ytest = ypred[target].values

            if model == 'pls':
                if LV == None:
                    print("For the 'pls' model, enter the desired number of latent variables in the LV argument.")
                    return None
                
                # For each number of latent variables from 1 to LV
                from sklearn.cross_decomposition import PLSRegression
                from sklearn.model_selection import cross_val_predict
                kern=None
                random_seed=None
                for lv in range(1, LV + 1):
                    pls = PLSRegression(n_components=lv, scale=False)
                    pls.fit(low_level_cal, ytrain)

                    # Pred
                    ycal_pred = pls.predict(low_level_cal).ravel()
                    # Cross-validation
                    y_cv = cross_val_predict(pls, low_level_cal, ytrain, cv=10)
                    ypred_pred = pls.predict(low_level_pred).ravel()

                    # Metrics for calibration
                    R2_cal = r2_score(ytrain, ycal_pred)
                    r2_cal = (np.corrcoef(ytrain, ycal_pred)[0, 1]) ** 2
                    rmse_cal = np.sqrt(mean_squared_error(ytrain, ycal_pred))
                    
                    # Cross-validation metrics
                    R2_cv = r2_score(ytrain, y_cv)
                    r2_cv = (np.corrcoef(ytrain, y_cv)[0, 1]) ** 2
                    rmsecv = np.sqrt(mean_squared_error(ytrain, y_cv))
                    bias_cv = sum(ytrain - y_cv)/ytrain.shape[0]
                    SDV_cv = (ytrain - y_cv) - bias_cv
                    SDV_cv = SDV_cv*SDV_cv
                    SDV_cv = np.sqrt(sum(SDV_cv)/(ytrain.shape[0] - 1))
                    tbias_cv = abs(bias_cv)*(np.sqrt(ytrain.shape[0])/SDV_cv)
                    rpd_cv = np.std(ytrain) / rmsecv if rmsecv != 0 else np.nan
                    rpiq_cv = iqr(ytrain, rng=(25, 75)) / rmsecv if rmsecv != 0 else np.nan

                    # Metrics for prediction
                    R2_pred = r2_score(ytest, ypred_pred)
                    r2_pred = (np.corrcoef(ytest, ypred_pred)[0, 1]) ** 2
                    rmse_pred = np.sqrt(mean_squared_error(ytest, ypred_pred))
                    bias_pred = sum(ytest - ypred_pred)/ytest.shape[0]
                    SDV_pred = (ytest - ypred_pred) - bias_pred
                    SDV_pred = SDV_pred*SDV_pred
                    SDV_pred = np.sqrt(sum(SDV_pred)/(ytest.shape[0] - 1))
                    tbias_pred = abs(bias_pred)*(np.sqrt(ytest.shape[0])/SDV_pred)
                    rpd = np.std(ytest) / rmse_pred if rmse_pred != 0 else np.nan
                    rpiq = iqr(ytest, rng=(25, 75)) / rmse_pred if rmse_pred != 0 else np.nan

                    # Create DataFrame with predictions
                    predicoes = pd.DataFrame({
                        'Results': np.concatenate((ycal_pred, ypred_pred)),
                        'Ref': np.concatenate((ytrain, ytest)),
                        'Set': ['cal'] * len(ytrain) + ['pred'] * len(ytest)
                    })

                    results[combinacoes_name][f"LV{lv}"] = {
                        'model': pls,
                        'coefficients': pls.coef_,
                        'predictions': predicoes,
                        'metrics': {
                            'R2 Cal': R2_cal,
                            'r2 Cal': r2_cal,
                            'RMSEC': rmse_cal,
                            'R2 CV': R2_cv,
                            'r2 CV': r2_cv,
                            'RMSECV': rmsecv,
                            'Bias CV': bias_cv,
                            'tbias CV': tbias_cv,
                            'RPD CV': rpd_cv,
                            'RPIQ CV': rpiq_cv,
                            'R2 Pred': R2_pred,
                            'r2 Pred': r2_pred,
                            'RMSEP': rmse_pred,
                            'Bias Pred': bias_pred,
                            'tbias Pred': tbias_pred,
                            'RPD Pred': rpd,
                            'RPIQ Pred': rpiq
                        }
                    }
            elif model == 'rf':
                if random_seed == None:
                    print("For the 'rf' model, enter the desired seed number in random_seed")
                    return None
                LV=None
                kern=None
                # Fit a Random Forest model with default hyperparameters
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(random_state=random_seed)
                rf.fit(low_level_cal, ytrain)

                # Pred
                ycal_pred = rf.predict(low_level_cal)
                ypred_pred = rf.predict(low_level_pred)

                # Metrics for calibration
                R2_cal = r2_score(ytrain, ycal_pred)
                r2_cal = (np.corrcoef(ytrain, ycal_pred)[0, 1]) ** 2
                rmse_cal = np.sqrt(mean_squared_error(ytrain, ycal_pred))

                # Metrics for prediction
                R2_pred = r2_score(ytest, ypred_pred)
                r2_pred = (np.corrcoef(ytest, ypred_pred)[0, 1]) ** 2
                bias_pred = sum(ytest - ypred_pred)/ytest.shape[0]
                SDV_pred = (ytest - ypred_pred) - bias_pred
                SDV_pred = SDV_pred*SDV_pred
                SDV_pred = np.sqrt(sum(SDV_pred)/(ytest.shape[0] - 1))
                tbias_pred = abs(bias_pred)*(np.sqrt(ytest.shape[0])/SDV_pred)    
                rmse_pred = np.sqrt(mean_squared_error(ytest, ypred_pred))
                rpd = np.std(ytest) / rmse_pred if rmse_pred != 0 else np.nan
                rpiq = iqr(ytest, rng=(25, 75)) / rmse_pred if rmse_pred != 0 else np.nan

                # Create DataFrame with predictions
                predicoes = pd.DataFrame({
                    'Results': np.concatenate((ycal_pred, ypred_pred)),
                    'Ref': np.concatenate((ytrain, ytest)),
                    'Set': ['cal'] * len(ytrain) + ['pred'] * len(ytest)
                })

                # For Random Forest there are no explicit coefficients or intercepts
                results[combinacoes_name]["RF"] = {
                    'model': rf,
                    'predictions': predicoes,
                    'metrics': {
                        'R2 Cal': R2_cal,
                        'r2 Cal': r2_cal,
                        'RMSEC': rmse_cal,
                        'R2 Pred': R2_pred,
                        'r2 Pred': r2_pred,
                        'RMSEP': rmse_pred,
                        'Bias Pred': bias_pred,
                        'tbias Pred': tbias_pred,
                        'RPD Pred': rpd,
                        'RPIQ Pred': rpiq
                    }
                }
            elif model == 'svm':
                if kern == None:
                    print("For the 'svm' model choose a kernel among 'linear', 'poly', 'rbf', 'sigmoid' using the kern argument")
                    return None
                LV=None 
                random_seed=None   
                from sklearn.svm import SVR
                svm = SVR(kernel=kern)
                svm.fit(low_level_cal, ytrain)

                # Pred
                ycal_pred = svm.predict(low_level_cal)
                ypred_pred = svm.predict(low_level_pred)

                # Metrics for calibration
                R2_cal = r2_score(ytrain, ycal_pred)
                r2_cal = (np.corrcoef(ytrain, ycal_pred)[0, 1]) ** 2
                rmse_cal = np.sqrt(mean_squared_error(ytrain, ycal_pred))

                # Metrics for prediction
                R2_pred = r2_score(ytest, ypred_pred)
                r2_pred = (np.corrcoef(ytest, ypred_pred)[0, 1]) ** 2
                rmse_pred = np.sqrt(mean_squared_error(ytest, ypred_pred))
                bias_pred = sum(ytest - ypred_pred)/ytest.shape[0]
                SDV_pred = (ytest - ypred_pred) - bias_pred
                SDV_pred = SDV_pred*SDV_pred
                SDV_pred = np.sqrt(sum(SDV_pred)/(ytest.shape[0] - 1))
                tbias_pred = abs(bias_pred)*(np.sqrt(ytest.shape[0])/SDV_pred)
                rpd = np.std(ytest) / rmse_pred if rmse_pred != 0 else np.nan
                rpiq = iqr(ytest, rng=(25, 75)) / rmse_pred if rmse_pred != 0 else np.nan

                # Create DataFrame with predictions
                predicoes = pd.DataFrame({
                    'Results': np.concatenate((ycal_pred, ypred_pred)),
                    'Ref': np.concatenate((ytrain, ytest)),
                    'Set': ['cal'] * len(ytrain) + ['pred'] * len(ytest)
                })

                # For SVM there are no explicit coefficients or intercepts
                results[combinacoes_name]["SVM"] = {
                    'model': svm,
                    'predictions': predicoes,
                    'metrics': {
                        'R2 Cal': R2_cal,
                        'r2 Cal': r2_cal,
                        'RMSEC': rmse_cal,
                        'R2 Pred': R2_pred,
                        'r2 Pred': r2_pred,
                        'RMSEP': rmse_pred,
                        'Bias Pred': bias_pred,
                        'tbias Pred': tbias_pred,
                        'RPD Pred': rpd,
                        'RPIQ Pred': rpiq
                    }
                }

    return results