import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
import scipy.stats as st

df = pd.read_csv(f"{os.path.join(os.path.dirname(__file__), '..' )}/data/derived/data.csv")
seed = 1234 

X = df.drop("area", axis=1)
y = df.area

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

## Function as seen here: https://stackoverflow.com/questions/37487830/how-to-find-probability-distribution-and-parameters-for-real-data-python-3
def get_best_distribution(data):
    dist_names = ["gamma", "exponweib", "weibull_max", "weibull_min", "expon", "lognorm", "beta"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

dist, p, params = get_best_distribution(y) # Gamma distribution
alpha, loc, scale = params

plt.figure(figsize=(8,5))
plt.hist(y, bins=30, density=True, alpha=0.6, label='Data', color="darkorange")
# Plot the gamma distribution density
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = st.gamma.pdf(x, a=alpha, loc=loc, scale=scale)
plt.plot(x, p, 'k', linewidth=1.5, label='Gamma Fit')
# Set plot title and labels
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.xlim(0, xmax)
plt.grid(alpha=.2)
plt.savefig(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/outputs/data_gamma_dist")
plt.show()


## GLM directly, frequentist
from sklearn.linear_model import GammaRegressor as gr
from sklearn.model_selection import KFold
X_train_cv = X_train.to_numpy()
y_train_cv = y_train.to_numpy() + 1e-4 # avoid zero-division, positive outcomes

## Find best penalty hyperparameter using cross-validation
alpha_grid = np.arange(0.1, 10, 0.1)
kf = KFold(n_splits=5)
alpha_mses = []
best_a, best_mse = None, np.inf
for a in alpha_grid:
    avg_mse = 0
    for i, (train_idcs, test_idcs) in enumerate(kf.split(X_train_cv)):
        gamma_glm = gr(alpha = a)
        gamma_glm.fit(X_train_cv[train_idcs, :], y_train_cv[train_idcs])
        avg_mse += mean_squared_error(y_train_cv[test_idcs], gamma_glm.predict(X_train_cv[test_idcs, :]))
    avg_mse /= 5
    alpha_mses.append(avg_mse)
    if avg_mse < best_mse:
        best_mse = avg_mse
        best_a = a

#np.save(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/src/saved_models/best_a.npy", best_a)
best_a = np.load(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/src/saved_models/best_a.npy")

plt.figure(figsize=(8,5))
plt.plot(alpha_grid, alpha_mses, color = "olivedrab", label="avg mse")
plt.axvline(best_a, color = "black", label=f"minimum: a = {best_a}", linestyle="--")
plt.grid(alpha=.2)
plt.xlabel("alpha")
plt.ylabel("Avg cross-validation MSE")
plt.legend()
plt.savefig(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/outputs/gridsearch_glm")
plt.show()

gamma_glm = gr(alpha = best_a.item())
gamma_glm.fit(X_train, y_train+1e-4)
print(f"MSE: {mean_squared_error(y_test+1e-4, gamma_glm.predict(X_test))}")
print(f"MAE: {mean_absolute_error(y_test+1e-4, gamma_glm.predict(X_test))}")

 
## Save model to same location as other models
import joblib
joblib.dump(gamma_glm, f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/gamma_glm.pkl")

""" Code below writes Bayesian regression model for the Gamma distribution, plots the coefficient estimates,
traceplots and histograms.

Moreover, fits 2 more models based on the distribution (Gamma and ZeroInflated Poisson) to the data

However, not very useful for inference, so this might be useful as resources on Bayesian regression with
Gamma-distributed outcomes are rare and incomplete online, so useful for modelling if confidence intervals
of training data itself and coefficients are of use
"""

"""
import pymc3 as pm
import arviz as az

## In GLM Gamma regression, assume alpha fixed bc mean is alpha*beta
## This means that variation in mean is due to beta, so relate beta to explanatory 
## variables. Use inverse link mu_i = (x_i beta)^{-1}


n_features = X.shape[1]
with pm.Model() as gamma_reg:
    # Priors
    # Adjusting the priors for the coefficients and sigma
    rate = pm.HalfNormal('rate')#, beta=10)
    intcpt = pm.Normal('int')#, mu=2, sigma=1)
    coefs = [pm.Normal(f'coef_{c}') for c in X_train.columns]
    X_model = pm.Data("X", X_train.to_numpy())
    # Deterministic for the mean
    
    # Likelihood
    likelihood = pm.Gamma('y', mu=pm.math.exp(intcpt + pm.math.dot(X_model, coefs)), alpha=1, beta=rate, observed=y_train+1e-4) # add 1e-4 to avoid zero-division
    
    # Inference
    trace = pm.sample(1000, tune=500, chains=2, return_inferencedata=True)
    pm.plot_trace(trace)

trace.to_netcdf(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/src/saved_models/trace_nc")
# Plotting the posterior distribution of the coefficients
pm.plot_posterior(trace, var_names=[f'coef_{c}' for c in X_train.columns])
plt.tight_layout()
plt.savefig(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/outputs/post_dist_coefs")
plt.show()

# Plotting for distr. of the intercept and rate
pm.plot_posterior(trace, var_names=["int", "rate"])
plt.tight_layout()
plt.savefig(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/outputs/post_dist_rate_int")
plt.show()


## Posterior predictions, already have trace from above
with gamma_reg:
    pm.set_data({"X": X_test})
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["y", "mu"], predictions=True)
    print(posterior_predictive)
    model_preds_constant = posterior_predictive["predictions_constant_data"]
    model_preds = posterior_predictive["predictions"]

model_preds.data_vars["mu"]
model_preds_constant.data_vars["X"]

mus = model_preds.data_vars["mu"].to_numpy()

import scipy.stats as st
st.gamma(loc=mus).pdf()
trace.posterior.data_vars["coef_DC"][:, 0].to_numpy()

predictions = np.zeros((len(X_test), 2000)) # first 500 for burn-in
def get_dot(s, row):
    coefs = np.zeros((6, len(X_test.columns)+1))
    coefs[:, 0] = trace.posterior.data_vars["int"][:, s].to_numpy()
    for j, c in enumerate(X_test.columns):
        coefs[:, j+1] = trace.posterior.data_vars[f"coef_{c}"][:, s].to_numpy()
    preds = np.exp(coefs[:, 0] + np.dot(row, coefs[:, 1:].T))
    return np.mean(preds)

# Calculate mean prediction across batches for each draw. Reset index for correct indexing
for i, r in X_test.reset_index().drop("index", axis = 1).iterrows():
    if i % 10 == 0: print(f"Iteration {i+1}/104")
    for s in range(500, 2500): # drop first to burn-in
        ## Have 6 chains per draw, put mean of those in predictions
        predictions[i, s-500] = get_dot(s, r)

np.save(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/src/saved_models/predictions_bayes", predictions)

## GLM directly, frequentist
from sklearn.linear_model import GammaRegressor as gr
gamma_glm = gr()
gamma_glm.fit(X_train, y_train+1e-5)
print(mean_squared_error(y_test+1e-5, gamma_glm.predict(X_test)))
print(mean_absolute_error(y_test+1e-5, gamma_glm.predict(X_test)))
print(mean_squared_error(y_test, np.mean(predictions, axis = 1)))

import statsmodels.api as sm
df_train = X_train.copy()
df_train["area"] = y_train
fml = f"area ~ X + Y + month + day + FFMC + DMC + DC + ISI + temp + RH + wind + rain"
mp = sm.GLM.from_formula(fml, family=sm.families.Gamma(link=sm.families.links.log()), data=df_train)
rp = mp.fit()
rp.summary()

mean_squared_error(y_test, rp.predict(X_test))
mean_absolute_error(y_test, rp.predict(X_test))

from statsmodels.discrete.count_model import ZeroInflatedPoisson as ZIP
zip = ZIP(y_train, X_train)#, exog_infl=X_train)
zip = zip.fit()
zip.summary()
mean_squared_error(y_test, zip.predict(X_test))
mean_absolute_error(y_test, zip.predict(X_test))
"""