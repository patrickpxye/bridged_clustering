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


def plotDensityGraph(result_bkm, result_knn, result_lin):

    kde = gaussian_kde(result_bkm)
    x = np.linspace(min(result_bkm), max(result_bkm), 2000)
    density = kde(x)
    plt.plot(x, density, label='Bridged Clustering')
    plt.fill_between(x, density, alpha=0.5)
    mean_bkm = np.mean(result_bkm)
    print("mean distance for bkm is ", mean_bkm)
    plt.axvline(mean_bkm, color='blue', linestyle='dashed', linewidth=1)

    kde = gaussian_kde(result_lin)
    x = np.linspace(min(result_lin), max(result_lin), 2000)
    density = kde(x)
    plt.plot(x, density, label='Linear Regression')
    plt.fill_between(x, density, alpha=0.5)
    mean_lin = np.mean(result_lin)
    print("mean distance for linear regression is ", mean_lin)
    plt.axvline(mean_lin, color='green', linestyle='dashed', linewidth=1)

    kde = gaussian_kde(result_knn)
    x = np.linspace(min(result_knn), max(result_knn), 2000)
    density = kde(x)
    plt.plot(x, density, label='KNN')
    plt.fill_between(x, density, alpha=0.5)
    mean_knn = np.mean(result_knn)
    print("mean distance for knn is ", mean_knn)
    plt.axvline(mean_knn, color='red', linestyle='dashed', linewidth=1)


    plt.xlabel('Euclidean Distance', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig('plots/fig.pdf')
    plt.show()
