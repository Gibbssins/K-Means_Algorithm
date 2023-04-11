import random
import numpy as np
from data_preprocessing import data_preprocessing

def k_mean_algo(n_centers ,data, iterations=1000, data_preprocess=False):

    if data_preprocess: data = data_preprocessing(data)

    # Random initialization of the centers
    """cent_min, cent_max = np.min(data, axis=0), np.max(data, axis=0)
    centroid_0 = np.random.uniform(cent_min, cent_max, size=(n_centers,cent_min.shape[0]))"""

    # Kmeans++
    n_cluster = n_centers
    first_cluster = random.choice(data)  # data[np.random.randint(data.shape[0])]
    lst_centroids = [first_cluster]
    for i in range(n_cluster -1):
        centroids = np.array(lst_centroids)
        p = np.min(np.linalg.norm(data[: ,None,
                                  :] - centroids[None ,: ,:],
                                  axis=2) ,axis=1 )**2
        new_centroid = data[np.random.choice(np.arange(data.shape[0]), p = p /np.sum(p))]
        lst_centroids.append(new_centroid)
    centroid_0 = np.array(lst_centroids)
    # print(centroid_0.shape)

    # label the data respect to the new cluster center
    label = np.argmin(np.linalg.norm(data[: ,None ,: ] -centroid_0[None ,: ,:], axis=2), axis=1)
    centroid_1 = np.zeros_like(centroid_0)
    for i in range(n_cluster):
        a = list(np.where(label==i)[0])
        if len(a )!=0:
            centroid_1[i] = np.mean(data[a], axis=0)
        else:
            centroid_1[i] = centroid_0[i]

    # initialization of the iteration counter
    itera = 0

    while not(np.allclose(centroid_0, centroid_1 ,0.003)) or itera <= iterations:

        # update the centroid
        centroid_0 = centroid_1
        # print(centroid_0.shape, data.shape)
        # update the knowledge of the label
        label = np.argmin(np.linalg.norm(data[: ,None ,:] - centroid_0[None ,: ,:], axis=2), axis=1)

        # recompute the centroids
        for i in range(n_cluster):
            a = list(np.where(label==i)[0])
            if len(a )!=0:
                centroid_1[i] = np.mean(data[a], axis=0)
            else:
                centroid_1[i] = centroid_0[i]
        itera +=1

    return centroid_1, label