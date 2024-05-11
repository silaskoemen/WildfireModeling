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
#plt.title('Gamma Distribution Fit')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.xlim(0, xmax)
plt.grid(alpha=.2)
plt.savefig(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/outputs/data_gamma_dist")
plt.show()

## Given that best-fitting distribution for the data is gamma, we can turn to a 
## Bayesian regression setting with Gamma prior on the regression coefficients (conjugate)
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
    #mu = pm.Deterministic('mu', pm.math.exp(intcpt + pm.math.dot(X_model, coefs)))
    
    # Likelihood
    likelihood = pm.Gamma('y', mu=pm.math.exp(intcpt + pm.math.dot(X_model, coefs)), alpha=1, beta=rate, observed=y_train+1e-4) # add 1e-4 to avoid zero-division
    
    # Inference
    trace = pm.sample(1000, tune=500, chains=2, return_inferencedata=True)
    pm.plot_trace(trace)

trace.to_netcdf("trace.nc")
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


## Could try to just sample normally after setting data
with gamma_reg:
    pm.set_data({"X": X_test})
    pred_samples = pm.sample()
pred_samples.shape

## Posterior predictions, already have trace from above
# Can just go through all model draws without burn in and find predictions

with gamma_reg:
    pm.set_data({"X": X_test})
    pp = pm.sample_posterior_predictive(trace, predictions=True, return_inferencedata=True, random_seed=seed)#, samples=500)
    #pp = pm.sample_posterior_predictive(trace, predictions=True, var_names=["y"])
pp.predictions
pp

y_preds = pp.predictions["y"].to_numpy()

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
"""trace.posterior.data_vars["coef_DC"][:, 0].to_numpy()

predictions = np.zeros((len(X_test), 2000)) # first 500 for burn-in
def get_dot(s, row):
    coefs = np.zeros((6, len(X_test.columns)+1))
    coefs[:, 0] = trace.posterior.data_vars["int"][:, s].to_numpy()
    for j, c in enumerate(X_test.columns):
        coefs[:, j+1] = trace.posterior.data_vars[f"coef_{c}"][:, s].to_numpy()
    #print(coefs, coefs.shape)
    preds = np.exp(coefs[:, 0] + np.dot(row, coefs[:, 1:].T))
    #print(preds, preds.shape)
    return np.mean(preds)

# Calculate mean prediction across batches for each draw. Reset index for correct indexing
for i, r in X_test.reset_index().drop("index", axis = 1).iterrows():
    if i % 10 == 0: print(f"Iteration {i+1}/104")
    for s in range(500, 2500): # drop first to burn-in
        ## Have 6 chains per draw, put mean of those in predictions
        predictions[i, s-500] = get_dot(s, r)

np.save("predictions_bayes", predictions)

## GLM directly, frequentist
from sklearn.linear_model import GammaRegressor as gr
gamma_glm = gr()
gamma_glm.fit(X_train, y_train+1e-5)
print(mean_squared_error(y_test+1e-5, gamma_glm.predict(X_test)))
print(mean_absolute_error(y_test+1e-5, gamma_glm.predict(X_test)))
print(mean_squared_error(y_test, np.mean(predictions, axis = 1)))"""

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