import numpy as np
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def identify_spec(label):
    q_index = label.find('Q')
    if q_index == -1 or q_index == len(label) - 1:
        return '/'
    # if label[q_index: q_index + 2] == 'QR' or label[q_index: q_index + 2] == 'QS':
    #     return 'QR/S'
    return label[q_index: q_index + 2]


def generate_pca_plot(data, species_labels, cluster_labels, unique_species, n_clusters):

    markers = ['o', 'x', '+', 's', 'd', '*', 'p', '^', '<', '>']
    species_to_marker = {species: markers[i % len(markers)] for i, species in enumerate(unique_species)}

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    plt.figure(figsize=(8, 5))
    for species, marker in species_to_marker.items():
        species_data = data_pca[species_labels == species]
        species_cluster_labels = cluster_labels[species_labels == species]

        missing_values = [i for i in range(n_clusters) if i not in species_cluster_labels]
        species_cluster_labels = np.append(species_cluster_labels, missing_values)
        species_data = np.vstack([species_data, np.zeros((len(missing_values), 2))])
        
        plt.scatter(species_data[:, 0], species_data[:, 1], c=species_cluster_labels, marker=marker, label=species, edgecolor='k', alpha=0.7, cmap='viridis')

    plt.title('Cluster Assignment by Species')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Species')
    plt.colorbar(label='Cluster')
    plt.show()


def generate_proportion_plot(species_labels, cluster_labels):

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({'species': species_labels, 'cluster': cluster_labels})
    cluster_counts = plot_data.groupby(['species', 'cluster']).size().unstack(fill_value=0)

    # Normalize the counts to get proportions
    cluster_proportions = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

    # Plotting
    cluster_proportions.plot(kind='bar', stacked=True, colormap='viridis', figsize=(5, 3))
    plt.title('Cluster Assignment by Species')
    plt.xlabel('Species')
    plt.ylabel('Cluster Assignments')
    plt.legend(title='Cluster', labels=[f'Cluster {x}' for x in range(1, 5)])
    plt.show()

    species_labels = species_labels.to_numpy()
    print("Adjusted Rand Index:", adjusted_rand_score(species_labels, cluster_labels))
    print("Normalized Mutual Information:", normalized_mutual_info_score(species_labels, cluster_labels))
