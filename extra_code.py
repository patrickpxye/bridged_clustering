# Use the actual species labels for evaluation
species_labels = df['spp']

# Calculate the Adjusted Rand Index and Normalized Mutual Information
ari = adjusted_rand_score(species_labels, cluster_labels)
nmi = normalized_mutual_info_score(species_labels, cluster_labels)

print("Adjusted Rand Index:", ari)
print("Normalized Mutual Information:", nmi)