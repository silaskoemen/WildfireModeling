import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1 import inset_locator
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as patches

df = pd.read_csv(f"{os.path.join(os.path.dirname(__file__), '..' )}/data/derived/data.csv")
seed = 1234 
X = df.drop("area", axis=1)
y = df.area
# Split into train, validation and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

xgb = XGBRegressor()
xgb.load_model(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/src/saved_models/xgb.txt")

def forecast(map=False, X=5, Y=5, month = 6, day=1, FFMC = 85, DMC = 50, DC = 100, ISI = 2, temp = 20, RH = 50, wind = 3, rain = 0):
    """ Function which either plots the forecast over the whole map (map=False) or for a specific
    (X, Y) pair and returns the predicted area
    """
    value_dict = locals()
    forecast_df = X_train.iloc[0, :].copy()
    forecast_df = pd.DataFrame(forecast_df).T
    for name, val in value_dict.items():
        if name != "map":
            forecast_df[name] = val
    if value_dict["map"] == False: # do not plot map, rather return value
        return np.exp(xgb.predict(forecast_df))-1
    else:
        # Y only has values from 2 through 9
        preds_fc = np.zeros((8, 9), dtype = np.float32)
        for x in range(1, 10):
            for y in range(2, 10):
                # As above could have full pass over all data, just setting X and Y
                forecast_df["X"] = x
                forecast_df["Y"] = y 
                preds_fc[y-2, x-1] = np.exp(xgb.predict(forecast_df)) - 1

        image_path = f"{os.path.join(os.path.dirname(__file__), '..')}/data/raw/montesinho_map.png"
        image = plt.imread(image_path)

        # Create a figure and axis - MEAN PREDICTIONS
        fig, ax = plt.subplots()
        ax.imshow(image, aspect='auto') # 'auto' adjusts the aspect ratio to match the image

        # Plot each point, scaling the size by the area value and color-coding
        for x in range(1, 10):
            for y in range(2, 10):
                # Color-code the dots
                color = cm.coolwarm(preds_fc[y-2, x-1] / np.max(preds_fc))
                if x == 1:
                    rect = patches.Rectangle((10, (y-1)*51.9*1.12), 85, 59, linewidth=.4, color = color, alpha = 0.4)
                else:
                    rect = patches.Rectangle(((x-1)*84.5*1.135, (y-1)*51.9*1.12), 94.6, 59, linewidth=.4, color = color, alpha = 0.4)
                ax.add_patch(rect)
        ax.axis('off')
        # Create an inset for the colorbar
        axins = inset_locator.inset_axes(ax, width="50%", height="4%", loc='lower left')
        norm = Normalize(vmin=np.min(preds_fc), vmax=np.max(preds_fc))
        cbar = plt.colorbar(cm.ScalarMappable(cmap='coolwarm', norm=norm), cax=axins, orientation='horizontal')
        cbar.set_label('Predicted ln(Area)')
        plt.show()

# Can use this to fill in any inputs, if map = True, whole map is plotted for input values (and all X,Y)
forecast(map=True, X = 5, Y = 5, month = 6, day = 1, FFMC=80, DMC = 50,
                 DC = 100, ISI = 2, temp = 10, RH = 30, wind = 10, rain = 10)

# Highest
forecast(map=True, X = 9, Y = 5, month = 9, day = 6, FFMC=86, DMC = 210,
                 DC = 320, ISI = 11, temp = 30, RH = 60, wind = 10, rain = 0)

# Lowest
forecast(map=True, X = 3, Y = 9, month = 1, day = 3, FFMC=92.5, DMC = 25,
                 DC = 600, ISI = 3, temp = 16, RH = 20, wind = 0, rain = 4)