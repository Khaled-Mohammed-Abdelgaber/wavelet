# -*- coding: utf-8 -*-
"""
Created on Fri May 20 23:53:48 2022

@author: Khaled
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.manifold import TSNE




data = pd.read_excel("train.xlsx")

#data = data /data.max(axis = 1)
data = data.loc[:,1:1375]





features = []
index = []


for i in range(2*1374):
    
    if i % 2 == 0:
        index.append("cA")  
        features.append(pywt.dwt(data.loc[:,int(i/2)+1], wavelet = "sym2")[0])
    else:
        index.append("cD")  
        features.append(pywt.dwt(data.loc[:,int(i/2)+1], wavelet = "sym2")[1])


train_sets = pd.DataFrame(data = (features[1::2]))

train_sets = train_sets.dropna(axis = 0 )

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(train_sets)
df = pd.DataFrame()
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
#df['pca-three'] = pca_result[:,2]

import seaborn as sns

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])


plt.figure(figsize=(16,10))
plt.scatter(x = pca_result[:,0],y= pca_result[:,1],cmap='viridis',c = range(len(pca_result[:,0])) )


plt.colorbar()
plt.show()

from sklearn.manifold import TSNE 
# Configure t-SNE function. 
embed = TSNE(
    n_components=2, # default=2, Dimension of the embedded space.
    perplexity=400, # default=30.0, The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
    early_exaggeration=12, # default=12.0, Controls how tight natural clusters in the original space are in the embedded space and how much space will be between them. 
    learning_rate=20, # default=200.0, The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If the learning rate is too high, the data may look like a ‘ball’ with any point approximately equidistant from its nearest neighbours. If the learning rate is too low, most points may look compressed in a dense cloud with few outliers.
    n_iter=5000, # default=1000, Maximum number of iterations for the optimization. Should be at least 250.
    n_iter_without_progress=300, # default=300, Maximum number of iterations without progress before we abort the optimization, used after 250 initial iterations with early exaggeration. 
    min_grad_norm=0.0000001, # default=1e-7, If the gradient norm is below this threshold, the optimization will be stopped.
    metric='euclidean', # default=’euclidean’, The metric to use when calculating distance between instances in a feature array.
    init='random', # {‘random’, ‘pca’} or ndarray of shape (n_samples, n_components), default=’random’. Initialization of embedding
    verbose=1, # default=0, Verbosity level.
    random_state=42, # RandomState instance or None, default=None. Determines the random number generator. Pass an int for reproducible results across multiple function calls.
    method='barnes_hut', # default=’barnes_hut’. By default the gradient calculation algorithm uses Barnes-Hut approximation running in O(NlogN) time. method=’exact’ will run on the slower, but exact, algorithm in O(N^2) time. The exact algorithm should be used when nearest-neighbor errors need to be better than 3%. 
    angle=0.5, # default=0.5, Only used if method=’barnes_hut’ This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
    n_jobs=-1, # default=None, The number of parallel jobs to run for neighbors search. -1 means using all processors. 
)

# Transform X
X_embedded = embed.fit_transform(train_sets)

plt.figure(figsize=(16,10))
plt.scatter(x = X_embedded[:,0],y= X_embedded[:,1],cmap='viridis',c = range(len(X_embedded[:,0])) )





