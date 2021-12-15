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
Unlike the linear regression, the response variable of lr is 0-1 binary variable, i.e. if an event will occur. The ideal is a unit step function $$P(y=1|x) = \begin{cases}
0, &if\ \beta_0 + \beta^TX < 0\\
0.5, &if\  \beta_0 + \beta^TX = 0\\
1, &if\ \beta_0 + \beta^TX > 0
\end{cases}$$ However, it's non-differentiable. So the commonly used substitution function is $$logit(P) = logit(odds) = ln(\frac{P}{(1-P}) = \beta_0 + \beta^TX$$ or $$P(y=1|x) = \frac{1}{1+e^{-(\beta_0 + \beta^TX)}}$$ That is, the log odds ratio is represented by a linear function of the input x. The closer the value of $\beta_0 + \beta^TX$ is to positive infinity, the closer the value of P(Y=1|x) probability is to 1. Therefore, the idea of logistic regression is to first fit a decision boundary (not limited to linear, but also polynomial) and then establish the probability link between this boundary and the classification, thus obtaining the probability in the binary case.  
Parameter estimation method: maximum likelihood estimation. $$L(\beta) = \prod{[p(xi)]^{yi}[1-p(xi)]^{1-yi}}$$ In the logistic regression model, maximizing the likelihood function and minimizing the loss function are equivalent. $$J(\beta) = -\frac{1}{n}\sum{lnL(\beta)}$$ Parameter solving methods: stochastic gradient descent(SGD) and Newton iterative method(Newton).  
Regulazition: L1(LASSO regression makes weights sparse, which is equivalent to adding to the model the prior knowledge that $\beta$ obeys a zero-mean Laplace distribution) or L2 paradigm(Ridge regression forces all $\beta$ as close to zero as possible but not to zero, which is equivalent to adding a priori knowledge to the model that $\beta$ obeys a zero-mean normal distribution).  
Lr can be employed to model the credit risk combined with the credit scoring model. The dimensions to be measured and their weight of importance will be different according to the specific target group. Common aspects including payment history, amount owned, new credit, types of credit, bad default records, etc.  
The credit score functions as a good tool in making decisions (eg. loan approval, credit card application) and determining the loan amount.  
The general process of building a credit scoring model follows four steps. First, input the binning variables. Second, build the lr model. Third, convert logistic regression coefficients into scores based on specified business parameters of "offset" and "factor". Finally, conduct model diagnosis and validation.  
To determine the score, we need to derive the values of "offset" and "factor" as $$Score = offset + factor * ln(odds).$$ For example, if we assume the score is 800 when odds equals to 50 and the pdo(points to double the odds) is 45, we can solve the problems through two equations $$800 = offset + factor \* ln(50)$$ $$800 + 45 = offset + factor \* ln(2\*50).$$
WOE: Weight of Evidence. Measure the importance of different bins, which varies in the same direction as the default rate. $$WOE = ln(\frac{yPctGood}{yPctBad})$$
IV: Information Value. Measure the importance of different variables. $$IV = \sum_{i=1}^{n}(yPctGood - yPctBad) * ln(\frac{yPctGood}{yPctBad})$$
### XgBoost(xgb)
The principle of xgb is gradient boosting decision tree(GBDT), which can be used in both the classification and the regression problems. Three basic elements of **boosting** are: (1) A loss function to be optimized; (2) A weak learner to make predictions; (3) An additive model. The **gradient boosting** adjusts the previous residual error when it adds a new weak learner, thus obtaining a higher accuracy. It's called 'gradient' because it uses the gradient descend to minimize the loss function. However, the implementation of gradient boosting is slow since each time it needs to construct a tree and add it to the model sequence. In terms of this problem, **xgb** owns an advantage of fast speed. What's more, it generally has a better performance than GBDT for the following reasons.  
(1) It adds L2 regulazition to the loss function, which lowers the model variance and prevents the overfitting problem.  
(2) It supports parallel computing on the feature level. Before training, xgb sorts the data in advance and stores the result in blocks, which enables to compute the feature information gain in parallel way.  
(3) Unlike the GBDT which uses the first-order derivative, it uses the second-order of Taylor spread expansion to approximate the loss function.  
(4) It handles missing values by auto matically learn the splitting directions of features with missing values.  
(5) It adopts feature subsampling. xgb draws on the random forest approach to support feature subsampling, which not only reduces overfitting but also reduces computation.  

### LightGBM(lgbm)
In principle, it is similar to GBDT and XGBoost in using the negative gradient of the loss function as the residual approximation of the current decision tree to fit the new decision tree.  
Compared to xgb, lgbm will perform better on several aspects. First, it uses the histogram optimization to map continuous values to discrete values, which reduces the memory consumption and compution cost compared to the pre-sorted algorithm in xgb. In addition, the histogram algorithm plays a role in regulazition. The less the number of bins, the higher risk of underfitting. Second, lgbm uses the leaf-wise learning other than level-wise learning to increase precision. Third, lgbm supports the input of original categorical features without pre-processing. In contrast, if we use one-hot coding to deal with the categorical feature, the information gain may be quite low due to the division imbalance.  
### Relevant Materials
Logistic Regression Credit Scoring: [LR_code.ipynb](https://github.com/sherryahaha/Risk-Models/blob/main/LRCredit/LR_code.ipynb)  
LightGBM: [LgbmCode.ipynb](https://github.com/sherryahaha/Risk-Models/blob/main/LgbmCredit/LgbmCode.ipynb)  
## Deep learning
- Neural Network
- LSTM
- Auto-Encoder
