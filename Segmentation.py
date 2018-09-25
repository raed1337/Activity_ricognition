import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import style
style.use('ggplot')



def importDataSegemntation():
    df_data = pd.read_csv("./output.csv",header=None)
    return df_data

def doKmeans():
    cluster = KMeans(n_clusters=12)
    data = importDataSegemntation()
    x_cols = data.columns[1:]
    data['cluster'] = cluster.fit_predict(data[data.columns[2:]])
    pca = PCA(n_components=2)
    data['x'] = pca.fit_transform(data[x_cols])[:, 0]
    data['y'] = pca.fit_transform(data[x_cols])[:, 1]
    activities_clusters = data.reset_index()
    return (activities_clusters)


data=doKmeans()
fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(data['x'],data['y'],c=data['cluster'],s=50)
ax.set_title('K-Means Clustering')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)
plt.show()

print(doKmeans())


"""ggplot(activities_clusters, aes(x='x', y='y', color='cluster')) + \
    geom_point(size=75) + \
    ggtitle("Customers Grouped by Cluster")"""





