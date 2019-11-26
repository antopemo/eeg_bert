############################################
#  Vector of Locally Aggregated EEGs       #
############################################
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np


def vlaeeg(data, num_cluster, norm='None', return_not_normed=False):
    kmeans = KMeans(num_cluster, random_state=0).fit(data)
    print(kmeans.labels_)
    vlaeeg_data = np.array()
    for i in range(len(kmeans.cluster_centers_)):
        vlaeeg_data[i] = np.sum(data[kmeans.labels_ == i] - kmeans.cluster_centers_[i], axis=0)
    if norm != 'None' and return_not_normed:
        return normalize(vlaeeg_data, norm=norm), vlaeeg_data
    elif norm != 'None' and not return_not_normed:
        return normalize(vlaeeg_data, norm=norm)
    else:
        return vlaeeg_data
