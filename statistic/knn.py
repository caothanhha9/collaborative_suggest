import pickle

import operator

from statistic import similarity_estimator
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os


def knn_total(x):
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(x)
    distances, indices = nbrs.kneighbors(x)
    return indices, distances


def knn_train(x, file_path=''):
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
    nbrs.fit(x)
    if (len(file_path) > 0) & (os.path.isfile(file_path)):
        pickle.dump(nbrs, open(file_path, 'rw'))
    return nbrs


def knn_predict(x, nbrs=None, file_path=''):
    if nbrs is not None:
        return nbrs.kneighbors(x)
    elif os.path.isfile(file_path):
        try:
            nbrs = pickle.load(open(file_path))
            return nbrs.kneighbors(x)
        except:
            return None
    else:
        return None


def similarity_knn(sample_id, x, k, threshold=0.0, dist_func=None):
    test_instance = x[sample_id]
    neighbor_indices = []
    neighbor_distances = []
    neighbors = []
    neighbors_indices_, neighbor_distances_, neighbors_ = get_sim_neighbors(
        x, test_instance, k + 1, threshold=threshold, dist_func=dist_func)
    for nb_id, nb_ in enumerate(neighbors_indices_):
        if not nb_ == sample_id:
            neighbor_indices.append(nb_)
            neighbor_distances.append(neighbor_distances_[nb_id])
            neighbors.append(neighbors_[nb_id])
    return neighbor_indices, neighbor_distances, neighbors


def get_sim_neighbors(training_set, test_instance, k, threshold=0.0, dist_func=None):
    distances = []
    if dist_func is None:
        dist_func = similarity_estimator.pearsonr
    for x in range(len(training_set)):
        # dist = similarity_estimator.pearsonr(test_instance, training_set[x])
        dist = dist_func(test_instance, training_set[x])
        if dist >= threshold:
            distances.append((x, dist, training_set[x]))
    distances.sort(key=operator.itemgetter(1), reverse=True)
    neighbors = []
    neighbor_indices = []
    neighbor_distances = []
    for x in range(k):
        if x < len(distances):
            neighbor_indices.append(distances[x][0])
            neighbor_distances.append(distances[x][1])
            neighbors.append(distances[x][2])
    return neighbor_indices, neighbor_distances, neighbors

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# X = np.array([[5.0, 4.0, 3.0, 2.0], [4.0, 4.0, 3.0, 1.0], [1.0, 2.0, 3.0, 4.0], [5.0, 4.0, 3.0, 3.0], [3.0, 3.0, 1.0, 2.0]])
# indices_, distances_ = knn_total(X)
# print indices_[0]
# print distances_[0]
# nbrs_ = knn_train(X)
# print nbrs_.kneighbors([[5.0, 4.0, 3.0, 2.0]])
# Y = np.array([[1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0, 0, 0]])
# print similarity_knn(0, X, k=5, threshold=0.8, dist_func=similarity_estimator.pearsonr)
# print similarity_knn(0, Y, k=5, threshold=0.8, dist_func=similarity_estimator.item_cosine_estimate)
