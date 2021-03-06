import numpy as np
import matplotlib.pyplot as plt
import itertools
import tqdm


def knearestneighbor(test_data, train_data, train_label, k):
    est_class = np.zeros(test_data.shape[-1])
    for i in range(test_data.shape[-1]):
        data_point = test_data[:, i]
        kth_label = np.int32(np.zeros(k))
        kth_distance = 100.0 * np.arange(1, k+1, 1)
        for j in range(train_label.size):
            cur_distance = np.sum((train_data[:, j] - data_point)**2)
            cur_max_idx = np.argmax(kth_distance)
            if cur_distance < kth_distance[cur_max_idx]:
                kth_distance[cur_max_idx] = cur_distance
                kth_label[cur_max_idx] = train_label[j]
        est_class[i] = np.argmax(np.bincount(kth_label))
    return est_class


def cross_validate(data, gt_labels, k, num_folds, feature_idx=None):
    fold_accuracies = np.zeros(num_folds)
    if feature_idx is not None:
        data = data[feature_idx, :]
        if len(data.shape) == 1:
            data = np.expand_dims(data, 0)
    conf_mat = np.zeros((num_folds, 5, 5))
    shuffle_indices = np.random.permutation(gt_labels.size)
    part = gt_labels.size/num_folds
    for i in range(num_folds):
        train_idx = np.concatenate((shuffle_indices[:int(i*part)], shuffle_indices[int((i+1)*part):]))
        test_idx = shuffle_indices[int(i*part):int((i+1)*part)]
        train_data, train_label = data[:, train_idx], labels[train_idx]
        test_data, test_label = data[:, test_idx], labels[test_idx]
        est_class = knearestneighbor(test_data, train_data, train_label, k)
        fold_accuracies[i] = np.where(est_class - test_label == 0)[0].size / test_label.size
        for j in range(conf_mat.shape[-1]):
            for l in range(conf_mat.shape[-1]):
                conf_mat[i, j, l] = np.where(est_class[test_label == j+1] == l+1)[0].size
    conf_mat = np.sum(conf_mat, axis=0)
    return np.mean(fold_accuracies), fold_accuracies, conf_mat


def select_features(data, labels, k, num_folds):
    sel_feature_ind = []
    acc = []
    for i in range(data.shape[0]):
        for feature_idx in tqdm.tqdm(itertools.combinations(range(data.shape[0]), i+1)):
            avg_acc, _, conf = cross_validate(data, labels, k, num_folds, feature_idx)
            sel_feature_ind.append(feature_idx)
            acc.append(avg_acc)
    return np.array(sel_feature_ind), np.array(acc)


def evaluate(data, labels):
    feature_idx = (1, 2, 3, 5, 6, 7, 9)
    accuracies = np.array([])
    conf_matrices = np.empty([0, 5, 5])
    for i in [1, 3, 7]:
        avg_accuracy, fold_accuracies, conf_mat = cross_validate(data, labels, i, 10, feature_idx)
        accuracies = np.append(accuracies, avg_accuracy)
        conf_matrices = np.concatenate((conf_matrices, np.expand_dims(conf_mat, 0)))
        print(f'Average accuracy of k={i} is {avg_accuracy*100:.2f}%.')
    return accuracies, conf_matrices


def kmeans_clustering(data, k):
    prev_centroids = np.ones((data.shape[0], k))
    centroids = data[:, np.random.choice(data.shape[-1], k)]
    c_labels = np.zeros(data.shape[-1])
    tol = 1e-60
    while np.mean((centroids - prev_centroids)**2) > tol:
        for i in range(data.shape[-1]):
            distance = np.zeros(k)
            for j in range(k):
                distance[j] = np.sqrt(np.sum((centroids[:, j] - data[:, i])**2))
            c_labels[i] = np.argmin(distance)
        prev_centroids = centroids.copy()
        for i in range(k):
            centroids[:, i] = np.mean(data[:, c_labels == i], axis=1)
    return c_labels, centroids


if __name__ == '__main__':
    feature_matrix = np.loadtxt(open('data/data.txt'))
    labels = np.int32(np.loadtxt(open('data/labels.txt')))
    feature_dict = ['Root Mean Square Mean', 'Zero Crossing Rate Mean', 'Spectral Centroid Mean',
                    'Spectral Flux Mean', 'Spectral Crest Mean', 'Root Mean Square Std', 'Zero Crossing Rate Std',
                    'Spectral Centroid Std', 'Spectral Flux Std', 'Spectral Crest Std']
    for i in range(10):
        avg_acc, acc, conf_mat = cross_validate(feature_matrix, labels, 3, 3, i)
        print(f'Average accuracy for {feature_dict[i]} is {avg_acc*100:.2f}%.')

    '''
    sel_feature_ind, accuracy = select_features(feature_matrix, labels, 3, 3)
    plt.plot(accuracy)
    plt.xlabel('Iterations')
    plt.ylabel('Average Accuracy')
    plt.show()
    best_combination = sel_feature_ind[np.argmax(accuracy)]
    print(best_combination)
    '''
    # accuracies, conf_matrices = evaluate(feature_matrix, labels)
    c_labels, centroids = kmeans_clustering(feature_matrix, 5)
    plt.plot(c_labels)
    plt.xlabel('Data Point')
    plt.ylabel('Cluster Label')
    plt.show()
    '''
    # Test scripts to prove my k-means clustering is right
    from sklearn.datasets import make_blobs

    # create dataset
    X, y = make_blobs(
        n_samples=150, n_features=2,
        centers=3, cluster_std=0.5,
        shuffle=True, random_state=0
    )

    plt.scatter(
        X[:, 0], X[:, 1],
        c='white', marker='o',
        edgecolor='black', s=50
    )
    plt.show()

    y_km, centroids = kmeans_clustering(X.T, 3)

    plt.scatter(
        X[y_km == 0, 0], X[y_km == 0, 1],
        s=50, c='lightgreen',
        marker='s', edgecolor='black',
        label='cluster 1'
    )

    plt.scatter(
        X[y_km == 1, 0], X[y_km == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='cluster 2'
    )

    plt.scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=50, c='lightblue',
        marker='v', edgecolor='black',
        label='cluster 3'
    )

    plt.scatter(
        centroids.T[:, 0], centroids.T[:, 1],
        s=250, marker='*',
        c='red', edgecolor='black',
        label='centroids'
    )
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()
    '''
    print
# EOF