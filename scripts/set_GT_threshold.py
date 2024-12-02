import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
    
df = pd.read_csv('...')


#filter based on size and DAPI intensity

plt.hist(df['nucleus_edu_area'], bins = 100, color='gray', alpha = 0.5)
plt.show()
df = df[df['nuclear_DAPI_area'] > 160]

plt.hist(df['nuclear_DAPI_intensity_mean'], bins = 100, color='gray', alpha = 0.5)
plt.show()
df = df[df['nuclear_DAPI_intensity_mean'] > 500]


#select the monoculture conditions for GT thresholding
df_class1 = df[df['target'] == 'astro']
df_class2 = df[df['target'] == 'SHSY5Y']

#plot the edu and brdu intensities
plt.scatter(df_class1['nucleus_edu_intensity_mean'], df_class1['nucleus_brdu_intensity_mean'], c='red', label = 'Astro', s = 0.1)
plt.scatter(df_class2['nucleus_edu_intensity_mean'], df_class2['nucleus_brdu_intensity_mean'], c='blue', label = 'SHSY5Y', s = 0.1)
plt.scatter(df['nucleus_edu_intensity_mean'], df['nucleus_brdu_intensity_mean'], c='gray', label = 'co-culture', s = 0.1)

thrx = 600
plt.axvline(thrx)
thry = 300
plt.axhline(thry)
plt.title('Scatterplot edu vs. brdu intensity')
plt.show()


# determine the true category based on the input threshold
c = []
for x, v in df.iterrows():
    if v["nucleus_edu_intensity_mean"] > thrx and v["nucleus_brdu_intensity_mean"] < thry:
        c.append("astro")
    elif v["nucleus_edu_intensity_mean"] < thrx and v["nucleus_brdu_intensity_mean"] > thry:
        c.append("SHSY5Y")
    else:
        c.append("inconclusive")
df["true_condition"] = c

df.to_csv('GT_data.csv', index=False)

