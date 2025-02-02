import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def decisionVector(sample, dim=3):
    
        # build an association matrix
        association_matrix = np.zeros((dim,dim))
        for i in range(dim):
            for j in range(dim):
                association_matrix[i,j] = np.sum((sample['morph_cluster'] == i) & (sample['gene_cluster'] == j))
    
        # Initialize decision array
        decision = np.zeros(dim, dtype=int)
    
        # Create a copy of the association matrix to manipulate
        temp_matrix = association_matrix.copy()
    
        # Fill the decision array
        for _ in range(dim):
            max_index = np.argmax(temp_matrix)
            max_location = np.unravel_index(max_index, temp_matrix.shape)
            decision[max_location[0]] = max_location[1]
            temp_matrix[max_location[0], :] = -np.inf
            temp_matrix[:, max_location[1]] = -np.inf
    
        return decision


def plotDensityGraph(result_bkm, result_knn_1, result_knn_2, result_knn_3, result_lin):

    kde = gaussian_kde(result_bkm)
    x = np.linspace(min(result_bkm), max(result_bkm), 2000)
    density = kde(x)
    plt.plot(x, density, label='Bridged Clustering')
    plt.fill_between(x, density, alpha=0.5)
    mean_bkm = np.mean(result_bkm)
    print("mean distance for bkm is ", mean_bkm)
    plt.axvline(mean_bkm, color='blue', linestyle='dashed', linewidth=1)

    # kde = gaussian_kde(result_lin)
    # x = np.linspace(min(result_lin), max(result_lin), 2000)
    # density = kde(x)
    # plt.plot(x, density, label='Linear Regression')
    # plt.fill_between(x, density, alpha=0.5)
    # mean_lin = np.mean(result_lin)
    # print("mean distance for linear regression is ", mean_lin)
    # plt.axvline(mean_lin, color='green', linestyle='dashed', linewidth=1)

    kde = gaussian_kde(result_knn_1)
    x = np.linspace(min(result_knn_1), max(result_knn_1), 2000)
    density = kde(x)
    plt.plot(x, density, label='KNN(k=1)')
    plt.fill_between(x, density, alpha=0.5)
    mean_knn = np.mean(result_knn_1)
    print("mean distance for KNN(k=1) is ", mean_knn)
    plt.axvline(mean_knn, color='red', linestyle='dashed', linewidth=1)

    kde = gaussian_kde(result_knn_2)
    x = np.linspace(min(result_knn_2), max(result_knn_2), 2000)
    density = kde(x)
    plt.plot(x, density, label='KNN(k=2)')
    plt.fill_between(x, density, alpha=0.5)
    mean_knn = np.mean(result_knn_2)
    print("mean distance for KNN(k=2) is ", mean_knn)
    plt.axvline(mean_knn, color='pink', linestyle='dashed', linewidth=1)

    kde = gaussian_kde(result_knn_3)
    x = np.linspace(min(result_knn_3), max(result_knn_3), 2000)
    density = kde(x)
    plt.plot(x, density, label='KNN(k=3)')
    plt.fill_between(x, density, alpha=0.5)
    mean_knn = np.mean(result_knn_3)
    print("mean distance for KNN(k=3) is ", mean_knn)
    plt.axvline(mean_knn, color='red', linestyle='dashed', linewidth=1)


    plt.xlabel('Euclidean Distance', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig('plots/5.pdf')
    plt.show()
