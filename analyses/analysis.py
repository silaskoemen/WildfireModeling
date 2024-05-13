import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
#import tensorflow as tf
import joblib
import sklearn, xgboost, matplotlib
print(sklearn.__version__, xgboost.__version__, matplotlib.__version__)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

df = pd.read_csv(f"{os.path.join(os.path.dirname(__file__), '..' )}/data/derived/data.csv")
seed = 1234 
X = df.drop("area", axis=1)
y = df.area
# Split into train, validation and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)


##### Load models with optimal hyperparameters #####
# IMPORTANT: especially joblib with sklearn might not work for different versions, see specific package 
# versions at /src/requirements.txt
xgb = XGBRegressor()
xgb.load_model(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/xgb.txt")
xgb_clas = XGBClassifier()
xgb_clas.load_model(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/xgb_clas.txt")
ols = LinearRegression()
ols = joblib.load(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/ols.pkl")
gp = GaussianProcessRegressor()
gp = joblib.load(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/gp.pkl")
rf = RandomForestRegressor()
rf = joblib.load(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/rf.pkl")

plt.figure(figsize=(7, 4), dpi = 100)
plt.bar(range(1, 13), 100*xgb.feature_importances_, color = "black", alpha = 0.7, label = "XGB")
plt.bar(range(1, 13), 100*rf.feature_importances_, color = "orange", alpha = 0.7, label = "RF")
plt.ylabel("Feature importance (%)")
plt.xticks(range(1, 13), labels=X_train.columns, rotation = 30)
plt.legend(loc = "upper left")
plt.savefig(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/outputs/FIs.png")
plt.show()

_, ax = plt.subplots(ncols=4, nrows=3, figsize=(12, 8), sharey=True, constrained_layout=True)
ax = ax.flatten()
for i in range(len(X.columns)):
    if i not in [0, 4, 8]:
        ax[i].set_ylabel("")
    PartialDependenceDisplay.from_estimator(xgb, X, features=[X.columns[i]], kind='both', ax = ax[i], ice_lines_kw={"color":"orange", "alpha":0.25}, pd_line_kw={"color":"crimson"})
plt.savefig(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/outputs/PDPs.png")
plt.show()

# Map predict size and probability for each square
mean_df = pd.DataFrame(X.mean()).T 

# Y only has values from 2 through 9
preds = np.zeros((8, 9), dtype = np.float32)
preds_prob = np.zeros((8, 9), dtype = np.float32)
for x in range(1, 10):
    for y in range(2, 10):
        # As above could have full pass over all data, just setting X and Y
        mean_df["X"] = x
        mean_df["Y"] = y 
        preds[y-2, x-1] = np.exp(xgb.predict(mean_df)) - 1
        preds_prob[y-2, x-1] = xgb_clas.predict_proba(mean_df)[0][-1]

## Instead of using mean (would keep predictions very similar), pass over whole dataset
## Just setting X and Y given
preds_pdp = np.zeros((8, 9), dtype = np.float32)
for x in range(1, 10):
    for y in range(2, 10):
        # As above could have full pass over all data, just setting X and Y
        xy_preds = []
        for i, r in X.iterrows():
            r = r.T
            r["X"] = x
            r["Y"] = y
            xy_preds.append(np.exp(xgb.predict(r)) - 1)
        preds_pdp[y-2, x-1] = np.mean(xy_preds)

#### Shades respective area with predicted size by XGB ####
from mpl_toolkits.axes_grid1 import inset_locator
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as patches

image_path = f"{os.path.join(os.path.dirname(__file__), '..')}/data/raw/montesinho_map.png"
image = plt.imread(image_path)

# Create a figure and axis - MEAN PREDICTIONS
fig, ax = plt.subplots()
ax.imshow(image, aspect='auto') # 'auto' adjusts the aspect ratio to match the image

# Plot each point, scaling the size by the area value and color-coding
for x in range(1, 10):
    for y in range(2, 10):
        size = 100 * preds[y-2, x-1]
        # Color-code the dots
        color = cm.coolwarm(preds[y-2, x-1] / np.max(preds))
        if x == 1:
            rect = patches.Rectangle((10, (y-1)*51.9*1.12), 83, 59, linewidth=.4, color = color, alpha = 0.4)
        else:
            rect = patches.Rectangle(((x-1)*84.5*1.135, (y-1)*51.9*1.12), 94.6, 59, linewidth=.4, color = color, alpha = 0.4)
        ax.add_patch(rect)
ax.axis('off')
# Create an inset for the colorbar
axins = inset_locator.inset_axes(ax, width="50%", height="4%", loc='lower left')
norm = Normalize(vmin=np.min(preds), vmax=np.max(preds))
cbar = plt.colorbar(cm.ScalarMappable(cmap='coolwarm', norm=norm), cax=axins, orientation='horizontal')
cbar.set_label('Area (hectares)')
# Save the figure
plt.savefig(f"{os.path.join(os.path.dirname(__file__), '..')}/outputs/pred_area_map.png")
plt.show()

# Create a figure and axis - FULL DATA PASS
fig, ax = plt.subplots()
ax.imshow(image, aspect='auto') # 'auto' adjusts the aspect ratio to match the image

# Plot each point, scaling the size by the area value and color-coding
for x in range(1, 10):
    for y in range(2, 10):
        size = 100 * preds_pdp[y-2, x-1]
        # Color-code the dots
        color = cm.coolwarm(preds[y-2, x-1] / np.max(preds_pdp))
        if x == 1:
            rect = patches.Rectangle((10, (y-1)*51.9*1.12), 84.8, 59, linewidth=.4, color = color, alpha = 0.4)
        else:
            rect = patches.Rectangle(((x-1)*84.5*1.137, (y-1)*51.9*1.12), 94.6, 59, linewidth=.4, color = color, alpha = 0.4)
        ax.add_patch(rect)
ax.axis('off')
# Create an inset for the colorbar
axins = inset_locator.inset_axes(ax, width="50%", height="4%", loc='lower left')
norm = Normalize(vmin=np.min(preds_pdp), vmax=np.max(preds_pdp))
cbar = plt.colorbar(cm.ScalarMappable(cmap='coolwarm', norm=norm), cax=axins, orientation='horizontal')
cbar.set_label('Area (hectares)')
# Save the figure
plt.savefig(f"{os.path.join(os.path.dirname(__file__), '..')}/outputs/pred_area_map_full.png")
plt.show()

### Plot predictions against true values ###
fig, ax = plt.subplots(1, 3, figsize=(11, 4))
ax = ax.flatten()
ax[0].scatter(y_test, xgb.predict(X_test), color = "black", label="XGB")
ax[0].set_xlabel("XGB predictions")
ax[0].set_ylabel("True values")
ax[0].legend()
ax[0].grid(alpha=.2) ###
ax[1].scatter(y_test, rf.predict(X_test), color = "forestgreen", label="RF")
ax[1].set_xlabel("RF predictions")
ax[1].legend()
ax[1].grid(alpha=.2) ###
ax[2].scatter(y_test, gp.predict(X_test), color = "red", label="GP")
ax[2].set_xlabel("GP predictions")
ax[2].legend()
ax[2].grid(alpha=.2) ###
plt.tight_layout()
plt.savefig(f"{os.path.join(os.path.dirname(__file__), '..')}/outputs/pred_vs_true")
plt.show()