import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss, accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(f"{os.path.join(os.path.dirname(__file__), '..' )}/data/derived/data.csv")
seed = 1234 
ols, gp, rf, xgb = LinearRegression(fit_intercept=True), GaussianProcessRegressor(random_state=seed, normalize_y=True), RandomForestRegressor(random_state=seed), XGBRegressor(random_state = seed)

X = df.drop("area", axis=1)
y = df.area
# Make classification model to predict whether there is a fire or not
y_clas = [1. if o > 0. else 0. for o in y]
y_clas = pd.Series(y_clas)

##### Optimal hyperparameters #####
# Split into train, validation and test set, always using same seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X, y, test_size=0.2, random_state=seed)
scaler = StandardScaler().fit(X_train_std)
X_train_std = scaler.transform(X_train_std)
X_test_std = scaler.transform(X_test_std)
# OLS doesn't have any tuneable parameters
ols.fit(pd.concat((X_train, X_val), axis = 0), pd.concat((y_train, y_val), axis = 0))
ols_norm = LinearRegression()
ols_norm.fit(X_train_std, y_train_std)

# Plot from normalized OLS model
plt.figure(figsize=(10, 5))
plt.bar(range(13), np.insert(ols_norm.coef_, 0, ols_norm.intercept_), color = "black")
plt.axhline(0, color = "black", linewidth = 0.5)
plt.ylabel("Coefficient")
plt.xticks(range(13), labels=np.insert(X_train.columns, 0, "Int"), fontsize=7, rotation=60)
plt.grid(alpha=.2)
plt.savefig(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/outputs/coefs_ols.png")
plt.show()

# Prediction quality identical so continue further with normal OLS model as it does not need a scaler
# that is only fitted on training data
print(mean_squared_error(y_test, ols.predict(X_test)), mean_squared_error(y_test, ols_norm.predict(X_test_std)))

from sklearn.feature_selection import f_regression
p_vals_ols = f_regression(X_train_std, y_train_std)[1]

for i, c in enumerate(ols_norm.coef_):
    print(f"{X.columns[i]}: {c}, pvalue {p_vals_ols[i]}")

# For latex table notation, use " & ".join(format(x, "10.4f") for x in ols_norm.coef_)

# GP can only decide over kernel, but leave at default
gp.fit(X = pd.concat((X_train, X_val), axis = 0), y = pd.concat((y_train, y_val), axis = 0))

# Ranfom forest optimal hyperparameters
pbounds_rf = {
    'n_estimators': (40, 600),
    'max_depth': (2, 7),
    'max_features': (0.5,0.99),
    'min_samples_split': (2, 10)
}
def test_fn_rf(n_estimators, max_depth, max_features, min_samples_split):
    """ Function to fit the model with given parameters and return the negative MSE on the validation set
    
    Args:
    n_estimators [float]: next optimal "guess" for number of estimators
    max_depth [float]: parameter for maximum depth
    max_features [float]: proportion of features to include in each split
    min_samples_split [float]: minimum samples per leaf needed for split
    
    Returns:
    [float]: negative MSE on validation set
    """
    # Round the float values to ints if expected by model constructor
    n_estimators = round(n_estimators)
    max_depth = round(max_depth)
    min_samples_split = round(min_samples_split)
    max_features = float(max_features)

    rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, max_features= max_features, n_jobs = 8)
    rf.fit(X_train, y_train)
    return -mean_squared_error(y_val, rf.predict(X_val))

rf_opt = BayesianOptimization(f = test_fn_rf, pbounds = pbounds_rf, random_state=seed)
rf_opt.maximize(n_iter = 100)
rf_dict = rf_opt.max

# Fit with best set of parameters
n_estimators = round(rf_dict["params"]["n_estimators"]) # 126
max_depth = round(rf_dict["params"]["max_depth"]) # 2
max_features = rf_dict["params"]["max_features"] # 0.7721
min_samples_split = round(rf_dict["params"]["min_samples_split"]) # 2
rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, max_features= max_features, n_jobs = 8)
rf.fit(pd.concat((X_train, X_val), axis = 0), pd.concat((y_train, y_val), axis = 0))

# XGBoost optimal hyperparameters based on BO
pbounds_xgb = {
    'n_estimators': (40, 600),
    'max_depth': (2, 7),
    'learning_rate': (0.001,0.1),
    'subsample': (0.5, 1)
}
def test_fn_xgb(n_estimators, max_depth, learning_rate, subsample):
    """ Function to fit the model with given parameters and return the negative MSE on the validation set
    
    Args:
    n_estimators [float]: next optimal "guess" for number of estimators
    max_depth [float]: parameter for maximum depth
    learning_rate [float]: learning rate used for XGB algorithm (weight of next tree)
    subsample [float]: subsampling proportion for training
    
    Returns:
    [float]: negative MSE on validation set
    """
    # Round the float values to ints if expected by model constructor
    n_estimators = round(n_estimators)
    max_depth = round(max_depth)
    xgb = XGBRegressor(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, subsample = subsample, n_jobs = 8)
    xgb.fit(X_train, y_train)
    return -mean_squared_error(y_val, xgb.predict(X_val))

xgb_opt = BayesianOptimization(f = test_fn_xgb, pbounds = pbounds_xgb, random_state=seed)
xgb_opt.maximize(n_iter = 100)
xgb_dict = xgb_opt.max

# Fit with best set of parameters
n_estimators = round(xgb_dict["params"]["n_estimators"]) # 279
max_depth = round(xgb_dict["params"]["max_depth"]) # 4
learning_rate = xgb_dict["params"]["learning_rate"] # 0.0047
subsample = xgb_dict["params"]["subsample"] # 0.6591
xgb = XGBRegressor(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, subsample = subsample)
xgb.fit(pd.concat((X_train, X_val), axis = 0), pd.concat((y_train, y_val), axis = 0))

##### Performance on test set #####
# Use MSE and MAE
mses, maes = [], []

# To avoid copy-pasting, use globals() to access models
names = ["ols", "gp", "rf", "xgb"]
for n in names:
    mses.append(mean_squared_error(y_test, globals()[n].predict(X_test)))
    maes.append(mean_absolute_error(y_test, globals()[n].predict(X_test)))
print(f"MSE values: {mses}")
print(f"MAE values: {maes}")

joblib.dump(ols, f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/ols.pkl")
joblib.dump(gp, f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/gp.pkl")
joblib.dump(rf, f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/rf.pkl")
xgb.save_model(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/xgb.txt")

##### XGB for classification #####
y_train_clas, y_test_clas = train_test_split(y_clas, test_size=0.2, random_state=seed)
y_train_clas, y_val_clas = train_test_split(y_train_clas, test_size=0.2, random_state=seed)

# Pbounds stay the same, just now classifier instead of regressor
def test_fn_xgb_clas(n_estimators, max_depth, learning_rate, subsample):
    n_estimators = round(n_estimators)
    max_depth = round(max_depth)
    xgb = XGBClassifier(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, subsample = subsample, n_jobs = 8, random_state = seed)
    xgb.fit(X_train, y_train_clas)
    return -log_loss(y_val_clas, xgb.predict_proba(X_val))

xgb_opt_clas = BayesianOptimization(f = test_fn_xgb_clas, pbounds = pbounds_xgb, random_state=seed)
xgb_opt_clas.maximize(n_iter = 100)
xgb_dict_clas = xgb_opt_clas.max

# Fit with best set of parameters
n_estimators = round(xgb_dict_clas["params"]["n_estimators"])
max_depth = round(xgb_dict_clas["params"]["max_depth"])
learning_rate = xgb_dict_clas["params"]["learning_rate"]
subsample = xgb_dict_clas["params"]["subsample"]
xgb_clas = XGBClassifier(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate, subsample = subsample)
xgb_clas.fit(pd.concat((X_train, X_val), axis = 0), pd.concat((y_train_clas, y_val_clas), axis = 0))

print(f"Test set accuracy XGBClassifier: {accuracy_score(y_test_clas, xgb_clas.predict(X_test))}")
xgb_clas.save_model(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/xgb_clas.txt")

##### Logistic regression for classification #####
from sklearn.linear_model import LogisticRegressionCV as LR
y_train_clas, y_test_clas = train_test_split(y_clas, test_size=0.2, random_state=seed)
logreg = LR(tol=1e-1, max_iter=150)
logreg.fit(pd.concat((X_train, X_val), axis = 0), y_train_clas)
print(accuracy_score(y_test_clas, logreg.predict(X_test))) # less than 50%, so regression more useful

"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# Neural network can use BO over num_layers, dropout, batch normalization and units
pbounds_nn = {
    'num_layers': (1, 4),
    'dropout': (0, 0.3),
    'batch_norm': (0,1),
    'units': (15, 200)
}
def test_fn_nn(num_layers, dropout, batch_norm, units):
    num_layers = round(num_layers)
    units = round(units)
    nn = Sequential()
    nn.add(Dense(units = units, activation = "relu"))
    if batch_norm > 0.5:
        nn.add(BatchNormalization())
    nn.add(Dropout(dropout))
    for _ in range(num_layers-1):
        nn.add(Dense(units = units, activation="relu"))
        if batch_norm > 0.5:
            nn.add(BatchNormalization())
        nn.add(Dropout(dropout))
    nn.add(Dense(units = 1, activation="relu"))
    nn.compile(optimizer="adam", loss = "mse")
    nn.fit(x = X_train, y = y_train, epochs = 100, batch_size = 64, verbose = 0)
    return -nn.evaluate(x = X_val, y = y_val, verbose = 0)

nn_opt = BayesianOptimization(f = test_fn_nn, pbounds = pbounds_nn, random_state=seed)
nn_opt.maximize(n_iter = 100)
nn_dict = nn_opt.max

# Fit with best set of parameters
nn = Sequential()
nn_units = round(nn_dict["params"]["units"])
nn_dropout = nn_dict["params"]["dropout"]
nn_batch_norm = True if nn_dict["params"]["batch_norm"] > 0.5 else False
nn_num_layers = round(nn_dict["params"]["num_layers"])

np.save(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/nn_params", np.array((nn_units, nn_dropout, nn_num_layers, nn_batch_norm)))
nn.add(Dense(units = nn_units, activation = "relu"))
if nn_batch_norm:
    nn.add(BatchNormalization())
nn.add(Dropout(nn_dropout))
for _ in range(nn_num_layers-1):
    nn.add(Dense(units = nn_units, activation="relu"))
    if nn_batch_norm:
        nn.add(BatchNormalization())
    nn.add(Dropout(nn_dropout))
nn.add(Dense(units = 1, activation="relu"))
nn.compile(optimizer="adam", loss = "mse")
nn.fit(x = pd.concat((X_train, X_val), axis = 0), y = pd.concat((y_train, y_val), axis = 0), epochs = 200, batch_size = 64)

nn.save(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/nn.keras")
"""