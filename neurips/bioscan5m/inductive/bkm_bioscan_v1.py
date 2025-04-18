import pandas as pd
import numpy as np
import random
import os
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from torchvision import models, transforms
from PIL import Image
from bioscan5m.scripts.utils import decisionVector, plotDensityGraph

# Step 1: Load the dataset
def load_dataset(csv_path, image_folder, n_families=5, n_samples=50):
    # Load the original CSV
    df = pd.read_csv(csv_path)
    
    # Select families with at least 50 samples
    families = df['family'].value_counts()
    selected_families = families[families >= n_samples].index.tolist()
    
    if len(selected_families) < n_families:
        raise ValueError("Not enough families with at least 50 samples.")
    
    selected_families = random.sample(selected_families, n_families)
    
    family_samples = []
    for family in selected_families:
        family_data = df[df['family'] == family]
        if len(family_data) >= n_samples:
            family_samples.append(family_data.sample(n=n_samples, random_state=42))
        else:
            print(f"Family {family} has less than 50 samples, skipping.")
    
    final_df = pd.concat(family_samples)
    
    # Images and barcodes loading
    images = {}
    for _, row in final_df.iterrows():
        images[row['processid']] = os.path.join(image_folder, f"{row['processid']}.jpg")
    
    return final_df, images

# Step 2: Load pre-trained models
def load_pretrained_models():
    # Load BarcodeBERT for genetic barcode encoding
    barcode_model_name = "bioscan-ml/BarcodeBERT"  # Replace with actual model path
    barcode_tokenizer = AutoTokenizer.from_pretrained(barcode_model_name, trust_remote_code = True)
    barcode_model = AutoModel.from_pretrained(barcode_model_name, trust_remote_code = True)

    # Load ResNet50 model for image encoding
    image_model = models.resnet50(pretrained=True)
    image_model.eval()

    # Define image preprocessing (e.g., resizing, normalization for ResNet50)
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return barcode_tokenizer, barcode_model, image_model, image_transform

# Step 3: Image and genetic barcode encoding
def encode_images(image_ids, image_folder, model, transform):
    features = []
    for processid in image_ids:
        image_path = image_folder.get(processid, None)
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                output = model(image)
            features.append(output.squeeze().numpy())
        else:
            print(f"Warning: Image {processid} not found or invalid!")
            features.append(np.zeros(model.fc.in_features))  # Placeholder for missing image
    return np.array(features)

def encode_genes(dna_barcodes, tokenizer, model):
    # Ensure dna_barcodes is a list of strings (each string represents a barcode)
    if isinstance(dna_barcodes, np.ndarray):
        dna_barcodes = [str(barcode) for barcode in dna_barcodes]  # Ensure it's a list of strings

    # Process each DNA barcode individually
    embeddings = []
    for barcode in dna_barcodes:
        encodings = tokenizer(barcode, return_tensors="pt", padding=True, truncation=True)

        # Ensure the tensor has the right shape (batch_size, seq_length)
        encodings = {key: value.unsqueeze(0) for key, value in encodings.items()}  # Add batch dimension
        
        with torch.no_grad():
            embedding = model(**encodings).last_hidden_state.mean(dim=1).numpy()  # Average across sequence length

        embeddings.append(embedding)

    embeddings = np.array(embeddings)

    # If the embeddings still have a dimension of size 1, flatten them to 2D
    if len(embeddings.shape) == 3:
        embeddings = embeddings.squeeze(1)  # Remove the singleton dimension

    return embeddings

def encode_images_for_samples(df, image_folder, image_model, image_transform):
    features = encode_images(df['processid'].values, image_folder, image_model, image_transform)
    df['morph_coordinates'] = features.tolist()  # Store image features in the dataframe
    return df

def encode_genes_for_samples(df, barcode_tokenizer, barcode_model):
    gene_features = encode_genes(df['dna_barcode'].values, barcode_tokenizer, barcode_model)
    df['gene_coordinates'] = gene_features.tolist()  # Store gene features in the dataframe
    return df

def knn_regression(supervised_df, test_df, n_neighbors=1, image_model=None, image_transform=None, barcode_tokenizer=None, barcode_model=None, image_folder=None):
    # Encode both supervised and test sets using the models
    supervised_df = encode_images_for_samples(supervised_df, image_folder, image_model, image_transform)
    supervised_df = encode_genes_for_samples(supervised_df, barcode_tokenizer, barcode_model)
    
    test_df = encode_images_for_samples(test_df, image_folder, image_model, image_transform)
    test_df = encode_genes_for_samples(test_df, barcode_tokenizer, barcode_model)
    
    # Extract features for supervised and test sets
    X_train = np.array(supervised_df['morph_coordinates'].tolist())
    y_train = np.array(supervised_df['gene_coordinates'].tolist())
    
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    # Test the model
    X_test = np.array(test_df['morph_coordinates'].tolist())
    predictions = knn.predict(X_test)
    
    # Calculate Euclidean distance between predicted and actual gene coordinates
    y_test = np.array(test_df['gene_coordinates'].tolist())
    distances = np.linalg.norm(predictions - y_test, axis=1)  # Euclidean distance
    
    return distances


# Calculate error between predicted and actual gene coordinates
def calculate_error(df, predicted_column, actual_column):
    # Extract predicted and actual gene coordinates (both should be vectors with shape (768,))
    predicted_gene_coords = np.array(df[predicted_column].tolist())
    actual_gene_coords = np.array(df[actual_column].tolist())

    # Ensure that both predicted and actual have the same shape
    if predicted_gene_coords.shape != actual_gene_coords.shape:
        print(f"Shape mismatch detected: Predicted shape = {predicted_gene_coords.shape}, Actual shape = {actual_gene_coords.shape}")
        raise ValueError("Shape mismatch between predicted and actual coordinates.")

    # Calculate Euclidean distance for each sample (row-wise distance)
    distances = np.linalg.norm(predicted_gene_coords - actual_gene_coords, axis=1)  # Euclidean distance for each sample

    # Return the mean and standard deviation of the distances
    return np.mean(distances), np.std(distances)


def run_experiment(csv_path, image_folder, n_families=5, n_samples=50):
    # Load dataset
    df, images = load_dataset(csv_path, image_folder, n_families, n_samples)

    # Split dataset: 20-20-7-3 split for each family
    final_data = []
    for family in df['family'].unique():
        family_data = df[df['family'] == family]
        family_data = family_data.sample(frac=1, random_state=42)
        
        image_samples = family_data.iloc[0:30]
        gene_samples = family_data.iloc[30:60]
        supervised_samples = family_data.iloc[60:70]
        inference_samples = family_data.iloc[70:100]
        
        final_data.append((image_samples, gene_samples, supervised_samples, inference_samples))
    
    # Flatten splits
    image_samples, gene_samples, supervised_samples, inference_samples = zip(*final_data)
    image_samples = pd.concat(image_samples)
    gene_samples = pd.concat(gene_samples)
    supervised_samples = pd.concat(supervised_samples)
    inference_samples = pd.concat(inference_samples)

    # Load pre-trained models
    barcode_tokenizer, barcode_model, image_model, image_transform = load_pretrained_models()

    # --- Step 3: Clustering ---
    # Use image_samples to form image clusters
    image_features = encode_images(image_samples['processid'].values, images, image_model, image_transform)
    image_kmeans = KMeans(n_clusters=n_families, random_state=42).fit(image_features)

    # Use gene_samples to form gene clusters
    gene_features = encode_genes(gene_samples['dna_barcode'].values, barcode_tokenizer, barcode_model)
    gene_kmeans = KMeans(n_clusters=n_families, random_state=42).fit(gene_features)

    # --- Step 4: Build decision vector using supervised_samples ---
    supervised_samples = supervised_samples.copy()  # avoid SettingWithCopyWarning
    # Assign image clusters to supervised samples
    sup_image_features = encode_images(supervised_samples['processid'].values, images, image_model, image_transform)
    supervised_samples['image_cluster'] = image_kmeans.predict(sup_image_features)
    # Assign gene clusters to supervised samples
    sup_gene_features = encode_genes(supervised_samples['dna_barcode'].values, barcode_tokenizer, barcode_model)
    supervised_samples['gene_cluster'] = gene_kmeans.predict(sup_gene_features)
    # Build the decision matrix using the supervised split only
    decision_matrix = decisionVector(supervised_samples, morph_column='image_cluster', gene_column='gene_cluster', dim=n_families)

    # --- Step 5: Perform inference using inference_samples only ---
    inference_samples = inference_samples.copy()
    # Get image clusters for inference samples
    inf_image_features = encode_images(inference_samples['processid'].values, images, image_model, image_transform)
    inference_samples['image_cluster'] = image_kmeans.predict(inf_image_features)
    # Map image clusters to predicted gene clusters using the decision matrix
    inference_samples['predicted_gene_cluster'] = inference_samples['image_cluster'].apply(lambda x: decision_matrix[x])
    
    # Compute centroids for gene clusters using gene_samples
    gene_samples = gene_samples.copy()
    gene_samples['gene_cluster'] = gene_kmeans.labels_
    gene_samples['gene_coordinates'] = gene_features.tolist()
    centroids = []
    for cluster in range(n_families):
        cluster_data = gene_samples[gene_samples['gene_cluster'] == cluster]
        if len(cluster_data) > 0:
            centroid = np.mean(np.stack(cluster_data['gene_coordinates'].values), axis=0)
        else:
            centroid = np.zeros(gene_features.shape[1])
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    # Assign predicted gene coordinates to inference samples based on predicted gene clusters
    inference_samples['predicted_gene_coordinates'] = inference_samples['predicted_gene_cluster'].apply(lambda x: centroids[x])
    
    # Get the actual gene coordinates for inference samples
    inf_gene_features = encode_genes(inference_samples['dna_barcode'].values, barcode_tokenizer, barcode_model)
    inference_samples['gene_coordinates'] = inf_gene_features.tolist()

    # --- Step 6: Compute error using only inference_samples ---
    bridged_error_mean, bridged_error_std = calculate_error(inference_samples, 'predicted_gene_coordinates', 'gene_coordinates')

    # --- Baseline KNN using supervised_samples and inference_samples ---
    knn_error_mean_1 = knn_regression(supervised_samples, inference_samples, n_neighbors=1, 
                                      image_model=image_model, image_transform=image_transform, 
                                      barcode_tokenizer=barcode_tokenizer, barcode_model=barcode_model, 
                                      image_folder=images)
    knn_error_mean_2 = knn_regression(supervised_samples, inference_samples, n_neighbors=2, 
                                      image_model=image_model, image_transform=image_transform, 
                                      barcode_tokenizer=barcode_tokenizer, barcode_model=barcode_model, 
                                      image_folder=images)
    knn_error_mean_3 = knn_regression(supervised_samples, inference_samples, n_neighbors=3, 
                                      image_model=image_model, image_transform=image_transform, 
                                      barcode_tokenizer=barcode_tokenizer, barcode_model=barcode_model, 
                                      image_folder=images)
    
    print(f"Bridged Clustering Error: Mean={bridged_error_mean}, Std={bridged_error_std}")
    print(f"KNN (k=1) Error: Mean={np.mean(knn_error_mean_1)}")
    print(f"KNN (k=2) Error: Mean={np.mean(knn_error_mean_2)}")
    print(f"KNN (k=3) Error: Mean={np.mean(knn_error_mean_3)}")




if __name__ == '__main__':
    # Path to the dataset and images
    csv_path = 'test_data.csv'  # Replace with actual CSV path
    image_folder = 'test_images'  # Replace with actual image folder
    
    # Run the experiment
    run_experiment(csv_path, image_folder, n_families=4, n_samples=100)