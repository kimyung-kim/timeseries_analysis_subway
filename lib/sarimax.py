import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

def mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)
    return np.mean(np.abs((actual - forecast) / actual)) * 100

def sarimax_result(train_data, test_data, arma_order, seasonal_ord, seasonality, exog_variables_train, exog_variables_valid, name):
    model = None
    if len(exog_variables_train) > 0:
        model = SARIMAX(train_data, order=(arma_order[0], arma_order[1], arma_order[2]), 
                    seasonal_order=(seasonal_ord[0], seasonal_ord[1], seasonal_ord[2], seasonality), exog=pd.concat(exog_variables_train,axis=1))
    else:
        model = SARIMAX(train_data, order=(arma_order[0], arma_order[1], arma_order[2]), 
                    seasonal_order=(seasonal_ord[0], seasonal_ord[1], seasonal_ord[2], seasonality))
    results = model.fit()
    
    train_rmse = np.sqrt(np.mean((np.maximum(results.predict()[5:], 0)-train_data[5:])**2))
    print("train RMSE : ", train_rmse)
    print("train MAPE : ", mape(train_data[5:], np.maximum(results.predict()[5:], 0)))

    forecast = None
    if len(exog_variables_train) > 0:
        forecast = results.forecast(steps=365, exog=pd.concat(exog_variables_valid, axis=1))
    else:
        forecast = results.forecast(steps=365)

    indices = np.argsort(-(np.maximum(forecast, 0)-test_data)**2)[10:]
    test_rmse = np.sqrt(np.mean((np.maximum(forecast, 0)-test_data)**2))
    adj_test_rmse = np.sqrt(np.mean((test_data[indices]- np.maximum(forecast[indices], 0))**2))
    print("adj test RMSE : ", adj_test_rmse)
    print("test RMSE : ", test_rmse)
    print("adj test MAPE : ", mape(test_data[indices], np.maximum(forecast[indices], 0)))
    print("test MAPE : ", mape(test_data, np.maximum(forecast, 0)))

    plt.figure(figsize=(10, 6))
    plt.plot(test_data, label='Original Data')
    plt.plot(pd.date_range('2023-01-01', periods=365, freq='D'), np.maximum(forecast, 0), label='Forecast')
    plt.title(name)
    plt.legend()
    plt.show()