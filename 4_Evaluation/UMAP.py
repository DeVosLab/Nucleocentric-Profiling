import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import umap  
import geopandas as gpd
import shapely.geometry
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


def z_score(df):
    normalized_df = (df-df.mean())/df.std()
    return (normalized_df)


################
## read in complete file
################

df = pd.read_csv('...')
df = df.groupby(["target"]).sample(n=int(len(df)*0.1), replace = False, random_state=0) ## sample equal number of instances per target group

df_label = df[['target','ROI_img_name','replicate']]
df_nmbrs = df.select_dtypes(['number'])
normalized_df = df_nmbrs.groupby(['replicate']).apply(lambda x: z_score(x)) ## normalize grouped per replicate
# normalized_df = df_nmbrs.apply(lambda x: z_score(x)) ## normalize not grouped

normalized_df = pd.concat([normalized_df, df_label], axis = 1)
normalized_df = normalized_df.dropna(axis = 0)
normalized_df = normalized_df[normalized_df.columns.drop(list(normalized_df.filter(regex='ratio|density|centroid|orientation|euler|replicate')))]
normalized_df.to_csv('combined_features_all.csv', index = False)

################
## Heatmap
################

normalized_df['well'] = normalized_df['ROI_img_name'].str[:-5]
df_label = normalized_df[['well', 'true_condition']]
df_label = df_label.drop_duplicates()
mean = normalized_df.groupby(['well']).mean()
mean = mean.reset_index()
mean = mean.merge(df_label, on = 'well')
mean = mean.drop(columns=['Unnamed: 0'])
mean = mean.drop(columns=['well'])
mean = mean[mean.columns.drop(list(mean.filter(regex='cyto|ratio|density|centroid|orientation|euler')))]
mean_cell = mean.filter(regex='cell') 
mean_nucleus = mean.filter(regex='nucleus') 


celltype = mean.pop("true_condition")
lut = {'astro':'magenta', 'SH-SY5Y':'green'}
row_colors = celltype.map(lut)
g = sns.clustermap(mean, row_colors=row_colors, xticklabels=True, yticklabels=False, cmap = 'mako')
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 10)
plt.show()


################
## UMAP
################

train_df = normalized_df.groupby(["target"]).sample(n=int(len(normalized_df)*0.4), replace = False, random_state=0) ## random sample to equalize groups
train_df.target = pd.Categorical(train_df.target)
train_df['code'] = train_df.target.cat.codes
train_df = train_df.dropna(axis = 1)
train_df = train_df.dropna(axis = 0)

## build umap
X, y = train_df.drop(["target",'code','ROI_img_name'], axis=1), train_df[['code']].values.flatten()
X = train_df.drop(['ROI_img_name','true_condition'], axis=1)
X = train_df
pipe = make_pipeline(SimpleImputer(strategy="mean"))
X = pipe.fit_transform(X.copy())
fit = umap.UMAP().fit(X) ## fit unsupervised UMAP
X_reduced = fit.transform(X)
X_reduced.shape


## plot UMAP
conditions = ['astro','SHSY5Y']
colors = {'astro':'magenta', 'SHSY5Y':'green'}
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
            s=15, facecolors = 'none', edgecolors='0.25',
            c=train_df.target.map(colors))
plt.xlabel("UMAP1", size=24)
plt.ylabel("UMAP2", size=24)
# add legend to the plot with names
plt.legend(handles=scatter.legend_elements()[2], 
           labels= conditions,
           title="Color code")
plt.tight_layout()
plt.show()


## Define polygons/regions of clusters
polygon_astro = shapely.geometry.Polygon(
    [
        (12,2.5),
        (12, 9),
        (19, 9),
        (19, 2.5)
    ]
)
polygon_SHSY5Y = shapely.geometry.Polygon(
    [
        (-15,-9.5),
        (-15, 0),
        (-7, 0),
        (-7, -9.5)
    ]
)

## plot UMAP with polygons
colors = {'mature':'gold', 'immature':'cyan'}
x,y = polygon_astro.exterior.xy
a,b = polygon_SHSY5Y.exterior.xy
plt.plot(x, y, label = "astro", c = 'magenta')
plt.plot(a, b, label = "SH-SY5Y", c = 'green')
conditions = ['astro', 'SH-SY5Y']
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
            s=15, facecolors = 'none', edgecolors='0.25',
            c=train_df.true_condition.map(colors))
plt.xlabel("UMAP1", size=24)
plt.ylabel("UMAP2", size=24)
# add legend to the plot with names
plt.legend(handles=scatter.legend_elements()[0], 
           labels=conditions,
           title="Color code")
plt.tight_layout()
plt.show()


### select points in polygon
points = gpd.GeoDataFrame(X_reduced, geometry=gpd.points_from_xy(X_reduced[:, 0], X_reduced[:, 1]))
points.reset_index()
points_within = []
for row in range(len(points)):
    point = points.iloc[row]['geometry']
    contains = point.within(polygon_astro)
    points_within.append(contains) 
train_df['predicted_astro'] = points_within


points_within = []
for row in range(len(points)):
    point = points.iloc[row]['geometry']
    contains = point.within(polygon_SHSY5Y)
    points_within.append(contains) 
train_df['predicted_SH-SY5Y'] = points_within

## make feature plots
train_df.shape
plt.figure(figsize=(20, 10))
plt.subplots_adjust(hspace=0.5)
plt.suptitle("feature importance", fontsize=18, y=0.95)

for (index, column) in enumerate(train_df):    ## plot all feature maps
    ax = plt.subplot(18, 15, index + 1)
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                s=0.00001,
                c=train_df[column], cmap = 'mako')
    ax.set_title(column.upper(), fontdict={'fontsize': 4, 'fontweight': 'medium'})
    ax.axis('off')
    ax.set_xlabel("")
    if index == 264: ## number of columns (features)
        break
plt.show()

scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
            s=0.1,
            c=train_df['nucleus_DAPI_circularity'], cmap = 'mako') ## plot single feature map
plt.show()

