# Solar-Energy-Forecasting
Solar Energy Forecasting

## Performance of Time Series Models vs Classical Machine Learning Model
Time series models are used for univariate analysis of the data, and these model make predictions on the basis of univariate analysis. Solar Energy Forecasting is not a univariate analysis, it depends on various meteorological coditions such as temperature, humidity, etc. So time series model such as AR, MA, ARMA, ARMIA are not performing well as compare to classical machine learning model.
If time series model such as AR, MA, ARMA, ARIMA performs well then more advance method such as RNN, LSTM, etc should also perform well.

## Order of Time Series
Moving Average : ARMA(0, 3)
ARMA : ARMA(8, 6)
ARIMA : ARIMA(5, 1, 4)

1. An ARIMA(0,1,0) model (or I(1) model) is given by {\displaystyle X_{t}=X_{t-1}+\varepsilon _{t}} X_{t}=X_{t-1}+\varepsilon _{t} — which is simply a random walk.
2. An ARIMA(0,1,0) with a constant, given by {\displaystyle X_{t}=c+X_{t-1}+\varepsilon _{t}} X_{t}=c+X_{t-1}+\varepsilon _{t} — which is a random walk with drift.
3. An ARIMA(0,0,0) model is a white noise model.
