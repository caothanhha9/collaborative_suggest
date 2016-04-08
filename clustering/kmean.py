from sklearn.cluster import KMeans


def cluster_train(k, x, max_iter=1000):
    new_kmean = KMeans(n_clusters=k, max_iter=max_iter)
    new_kmean.fit_predict(x)
    return new_kmean


def cluster_predict(x, kmean_model):
    assigned_labels = kmean_model.predict(x)
    return assigned_labels

# k_ = 2
# X = [[5.0, 4.0, 3.0], [5.0, 5.0, 3.0], [1.0, 2.0, 4.0], [2.0, 3.0, 4.0]]
#
# test_kmean = cluster_train(k_, X)
#
# x_test = [[4.0, 4.0, 3.0], [1.0, 2.0, 3.0]]
# print cluster_predict(x_test, test_kmean)
