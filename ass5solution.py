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
            avg_acc, _, conf = cross_validate(feature_matrix, labels, k, num_folds, feature_idx)
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
    return


if __name__ == '__main__':
    feature_matrix = np.loadtxt(open('data/data.txt'))
    labels = np.int32(np.loadtxt(open('data/labels.txt')))
    '''
    sel_feature_ind, accuracy = select_features(feature_matrix, labels, 3, 3)
    plt.plot(accuracy)
    plt.xlabel('Iterations')
    plt.ylabel('Average Accuracy')
    plt.show()
    best_combination = sel_feature_ind[np.argmax(accuracy)]
    print(best_combination)
    '''
    accuracies, conf_matrices = evaluate(feature_matrix, labels)
    print
# EOF