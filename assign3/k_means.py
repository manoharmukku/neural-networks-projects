# Author: Manohar Mukku
# Date: 22.08.2018
# Desc: K-means clustering implementation

class k_means_clustering(object):
    def __init__(self):
        return None

    def get_cluster_centroids(self, n_clusters, input_data):
        self.K = n_clusters
        self.data = input_data
        self.n_features = input_data.shape[1]
        self.n_samples = input_data.shape[0]
        self.assigned_centroids = np.empty(shape=n_samples, dtype=int)
        self.centroids = np.empty(shape=(self.K, self.n_features), dtype=float)

        initialize_centroids()

        while (1):
            assign_centroids()

            prev_centroids = np.copy(self.centroids)
            update_centroids()

            if (np.array_equal(prev_centroids, self.centroids)):
                break

        return self.centroids

    def get_min_centroid(self, row):
        distance = (self.centroids - row)**2
        distance = np.sum(distance, axis = 1)
        distance = np.sqrt(distance)

        return np.argmin(distance)

    def assign_centroids(self):
        for i, row in enumerate(self.data):
            assigned_centroids[i] = get_min_centroid(row)

    def update_centroids(self):
        assigned_points = [[] for i in range(self.K)]

        for i, row in enumerate(self.data):
            assigned_points[self.assigned_centroids[i]].append(row)

        for i in range(self.K):
            points = np.asarray(assigned_points[i])
            self.centroids[i] = np.copy(np.mean(points, axis = 0))
