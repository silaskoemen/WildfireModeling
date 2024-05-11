import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

"""
File to process the raw data stored under data/raw/firestfires.csv, from https://osf.io/6a5hy/

The columns month and day are categorical and are transformed to integers and the outcome variable is transformed
"""

data = pd.read_csv(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/data/raw/forestfires.csv")
plt.hist(data.area, bins=50)
plt.xlabel("Area")
plt.ylabel("Count")
plt.grid(alpha=.2)
plt.savefig(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/outputs/hist_orig.png")
plt.show()
# Categorical to numerical
month_dict = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
day_dict = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}

for i in range(len(data)):
    data.loc[i, "month"] = month_dict[data.loc[i, "month"]]
    data.loc[i, "day"] = day_dict[data.loc[i, "day"]]

# Transform target variable
data.area = np.log(data.area + 1)
plt.hist(data.area, bins=50)
plt.xlabel("Area")
plt.ylabel("Count")
plt.grid(alpha=.2)
plt.savefig(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/outputs/hist_transformed.png")
plt.show()
# Save dataframe
data.to_csv(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/data/derived/data.csv", index=False)
