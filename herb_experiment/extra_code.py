# Use the actual species labels for evaluation
species_labels = df['spp']

# Calculate the Adjusted Rand Index and Normalized Mutual Information
ari = adjusted_rand_score(species_labels, cluster_labels)
nmi = normalized_mutual_info_score(species_labels, cluster_labels)

print("Adjusted Rand Index:", ari)
print("Normalized Mutual Information:", nmi)

score = 0
k = 300
for j in range(k):
    decision_matrix = np.zeros((2, 2))
    for i in range(135):
        supervised_pt = df.sample(1)
        #keep only 16 columns of supervised data
        gmm_pt = supervised_pt[["Lobe.number","BL","PL","BW","TLIW","TLL","TLDW","TEL","BLL","LLL","BSR","LSR","LLDW","LLIW","MidVeinD","BL_PL"]].dropna()
        kmeans_pt = supervised_pt[["Latitude", "Longitude","LLL.LLLDW","BL.BW"]].dropna()
        pred1 = gmm.predict(scaler.transform(transformer.transform(gmm_pt)))
        pred2 = loc_gmm.predict(scaler_2.transform(transformer_2.transform(kmeans_pt)))
        decision_matrix[pred1[0]][pred2[0]] += 1

    #print("Decision matrix: ")
    #print(decision_matrix)

    decision_vector = np.argmax(decision_matrix, axis=1)
    #if the decision vector does not sum up to 1, generate a random vector that contains 0 and 1
    if sum(decision_vector) == 0:
        decision_vector[np.random.randint(2)] = 1
    elif sum(decision_vector) == 2:
        decision_vector[np.random.randint(2)] = 0
    #print("Decision vector: ", decision_vector)
    if decision_vector[0] == 1 and decision_vector[1] == 0:
        score = score + 1
    
print("Accuracy: ", score/k)

x = np.array([5, 10, 20, 30, 40, 50, 70, 100])
y = np.array([0.714, 0.768, 0.834, 0.870, 0.894, 0.894, 0.938, 0.966])
plt.plot(x, y)
plt.xlabel("Number of supervised data points")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of supervised data points")