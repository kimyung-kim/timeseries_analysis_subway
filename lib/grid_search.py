import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def find_best_parameter(train_data, test_data, exog_variables):
    candidate = [[p, d, q, s_p, s_d, s_q, s] for p in range(6) for d in range(1, 2) for q in range(6) 
                    for s_p in range(3) for s_d in range(1, 2) for s_q in range(3) for s in [7]]
    best_parameter = [0, 0, 0, 0, 0, 0, 0]
    rmse = 100000000
    cnt = 0
    for p, d, q, s_p, s_d, s_q, s in candidate:
        try:
            model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(s_p, s_d, s_q, s), exog=pd.concat(exog_variables[0],axis=1))
            results = model.fit()
            forecast = results.forecast(steps=365, exog=pd.concat(exog_variables[1],axis=1))
            test_rmse = np.sqrt(np.mean((np.maximum(forecast, 0)-test_data)**2))
            if rmse > test_rmse:
                rmse = test_rmse
                best_parameter = [p, d, q, s_p, s_d, s_q, s]
            cnt = cnt + 1
            if cnt % 50 == 0:
                print(cnt, "models checked")
        except:
            pass
    
    return best_parameter