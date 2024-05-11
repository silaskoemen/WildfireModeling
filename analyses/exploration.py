import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

df = pd.read_csv(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/data/derived/data.csv")
colors = ("#FC8A17", "#F73718") # prob after have all put in src as list

## Create covariance matrix for all features, could use for sampling
# from sklearn.preprocessing import MinMaxScaler
# norm_df = MinMaxScaler().fit_transform(df)
# norm_df = pd.DataFrame(norm_df, columns = df.columns)
# cov = np.cov(norm_df.drop("area", axis = 1), rowvar=False)
# cov_df = pd.DataFrame(cov, columns=df.columns[:-1])
# cov_df.index = df.columns[:-1]
# https://www.statsmodels.org/stable/generated/statsmodels.discrete.count_model.ZeroInflatedPoisson.html
# could model as zero inflated Poisson from rest
# could then generate more from there from data distr.

# Scatterplots against area for some early insights
for n in df.columns[:-1]: # without last to drop area
    plt.scatter(df[n], df.area, c=colors[0])
    plt.ylabel("log Area")
    plt.xlabel(n)
    plt.savefig(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/outputs/scatter_area_{n}.png")

# Area/Incidents over map
image_path = f"{os.path.join(os.path.dirname( __file__ ), '..' )}/data/raw/montesinho_map.png"
image = plt.imread(image_path)

fig, ax = plt.subplots()

# Display the image with the correct aspect ratio
ax.imshow(image, aspect='auto') # 'auto' adjusts the aspect ratio to match the image

# Plot each point, scaling the size by the area value
for index, row in df.iterrows():
    # Normalize the area to a suitable size for the plot
    size = row['area']*1.5
    if row["X"] == 1:
        ax.scatter(70+np.random.normal(0, 8), (row['Y']-1)*51.9*1.13+25.95+np.random.normal(0, 6), s=size, color=colors[1], alpha=0.6)
    else:
        ax.scatter((row['X']-1)*84.5*1.15+42.25+np.random.normal(0, 8), (row['Y']-1)*51.9*1.13+25.95+np.random.normal(0, 6), s=size, color=colors[1], alpha=0.6)

# Hide the axis
ax.axis('off')

# Show the plot
plt.savefig(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/outputs/map_fires.png")
plt.show()