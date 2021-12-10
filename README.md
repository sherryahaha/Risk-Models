# Risk Models
## Statistical models
The common usage of statistical models focuses on the analysis of price yield trend, volatility and risk. Some time series models such as ARIMA-GARCH and risk estimation methods such as Value-at-Risk will be useful in this part.
### ARIMA-GARCH  
ARIMA models, i.e., summated Autoregressive Moving Average models, are mainly used to fit differential stationary series. The ARIMA (p, d, q) model represents that a series after the d-order differencing process is stationary, and the ARMA (p, q) model can be further fitted to it.

Some series, although passing the stationarity test, will exhibit sharp fluctuations at certain periods. The clustering effects have brought inconvenience to the prediction of time series in macroeconomic and financial areas, such as interest rates and stock prices. GARCH models are used to model volatility assuming a symmetric effect (if it is asymmetric TGARCH, EGARCH or GJR can be used).  
### Value-at-Risk  
According to the definition on the CFA website, Value-at-Risk (VaR) is the expected maximum amount of loss over a certain holding period under normal market fluctuation.
### Relevant Materials  
[Yilin Wu, Shiyu Ma. Impact of COVID-19 on energy prices and main macroeconomic indicators—evidence from China's energy market[J]. Green Finance, 2021, 3(4): 383-402. doi: 10.3934/GF.2021019](https://www.aimspress.com/article/doi/10.3934/GF.2021019)
## Machine learning
Predictive models: Predict users' future behavior based on their historical information
### Logistic Regression(lr)
Unlike the linear regression, the response variable of lr is 0-1 binary variable, i.e. if an event will occur. The formula is $$logit(P) = logit(odds) = ln(\frac{P}{(1-P}) = \beta_0 + \beta^TX$$ or $$P(y=1|x) = \frac{1}{1+e^{-(\beta_0 + \beta^TX)}}$$
Lr can be employed to model the credit risk combined with the credit scoring model. The dimensions to be measured and their weight of importance will be different according to the specific target group. Common aspects including payment history, amount owned, new credit, types of credit, bad default records, etc.  
The credit score functions as a good tool in making decisions (eg. loan approval, credit card application) and determining the loan amount.  
The general process of building a credit scoring model follows four steps. First, input the binning variables. Second, build the lr model. Third, convert logistic regression coefficients into scores based on specified business parameters of "offset" and "factor". Finally, conduct model diagnosis and validation.  
To determine the score, we need to derive the values of "offset" and "factor" as $$Score = offset + factor * ln(odds).$$ For example, if we assume the score is 800 when odds equals to 50 and the pdo(points to double the odds) is 45, we can solve the problems through two equations $$Score = 800 = offset + factor*ln(50); 800 + 45 = offset + factor * ln(2*50).$$
WOE: Weight of Evidence. Measure the importance of different bins, which varies in the same direction as the default rate. $$WOE = ln(\frac{yPctGood}{yPctBad})$$
IV: Information Value. Measure the importance of different variables. $$IV = \sum_{i=1}^{n}(yPctGood - yPctBad) * ln(\frac{yPctGood}{yPctBad})$$
### XgBoost(xgb)
### LightGBM(lgbm)

### Relevant Materials

## Deep learning
- Neural Network
- LSTM
- Auto-Encoder
