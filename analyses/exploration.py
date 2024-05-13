import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

df = pd.read_csv(f"{os.path.join( os.path.dirname( __file__ ), '..' )}/data/derived/data.csv")
colors = ("#FC8A17", "#F73718")

# Create heatmap with correlation values, similar to: https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
plt.figure(figsize=(12, 6))
# mask for upper triangle
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.savefig(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/outputs/corr_heatmap.png")
plt.show()

# Scatterplots against area for some early insights
for n in df.columns[:-1]: # without last to drop area
    plt.scatter(df[n], df.area, c=colors[0])
    plt.ylabel("log Area")
    plt.xlabel(n)
    plt.savefig(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/outputs/scatter_area_{n}.png")

# Scatterplots all in one image
fig, ax = plt.subplots(3, 4, figsize=(13, 8))
ax = ax.flatten()
for i, n in enumerate(df.columns[:-1]): # without last to drop area
    ax[i].scatter(df[n], df.area, c=colors[0], label = n)
    ax[i].set_ylabel("log Area")
    ax[i].set_xlabel(n)
    ax[i].legend(loc="upper right")
    ax[i].grid(alpha=.2)
plt.tight_layout()
plt.savefig(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/outputs/all_scatters.png")
plt.show()

# Area/Incidents over map
image_path = f"{os.path.join(os.path.dirname( __file__ ), '..' )}/data/raw/montesinho_map.png"
image = plt.imread(image_path)

fig, ax = plt.subplots()
ax.imshow(image, aspect='auto') # 'auto' adjusts the aspect ratio to match the image

for index, row in df.iterrows():
    # Normalize the area to a suitable size for the plot
    size = row['area']*1.5
    if row["X"] == 1:
        ax.scatter(70+np.random.normal(0, 8), (row['Y']-1)*51.9*1.13+25.95+np.random.normal(0, 6), s=size, color=colors[1], alpha=0.6)
    else:
        ax.scatter((row['X']-1)*84.5*1.15+42.25+np.random.normal(0, 8), (row['Y']-1)*51.9*1.13+25.95+np.random.normal(0, 6), s=size, color=colors[1], alpha=0.6)

ax.axis('off')
plt.savefig(f"{os.path.join(os.path.dirname( __file__ ), '..' )}/outputs/map_fires.png")
plt.show()