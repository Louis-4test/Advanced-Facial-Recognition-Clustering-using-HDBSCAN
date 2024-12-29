import matplotlib
matplotlib.use('Agg')  
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
from hdbscan import HDBSCAN 
from sklearn.metrics import pairwise_distances

# Path to the directory containing folders with images
base_directory = '/home/fola/Downloads/B.Tech/AI/code4/lfw'

# Function to lazily load images
def lazy_load_images(base_directory, max_images=100):
    image_count = 0
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    img = Image.open(file_path).convert('RGB')
                    img = img.resize((100, 100))  
                    yield np.array(img), folder_name
                    image_count += 1
                    if image_count >= max_images:
                        return

# Load images
max_images_to_load = 100
images = []
for img, _ in lazy_load_images(base_directory, max_images=max_images_to_load):
    images.append(img)

images = np.array(images)

# Function to introduce noise
def add_noise(images, noise_factor=0.1):
    noisy_images = images + noise_factor * np.random.randn(*images.shape)
    noisy_images = np.clip(noisy_images, 0, 1)  # Clipping to maintain pixel value range
    return noisy_images

# Add noise to images
noisy_images = add_noise(images)
num_images, height, width, channels = noisy_images.shape

# Flatten the images
flattened_images = noisy_images.reshape(num_images, height * width * channels)

# Standardize the dataset
scaler = StandardScaler()
standardized_images = scaler.fit_transform(flattened_images)

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# PCA Reduction
pca = PCA(n_components=0.90)  
reduced_images = pca.fit_transform(standardized_images)

# Visualize PCA Reduced Data
plt.scatter(reduced_images[:, 0], reduced_images[:, 1], alpha=0.5)
plt.title('PCA Reduced Data Visualization (Noisy Images)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.savefig('pca_reduced_data_noisy.png')  
plt.close()

# Function to apply HDBSCAN with specified distance metric
def perform_hdbscan_clustering(data, min_cluster_size, min_samples, metric):
    if metric == 'cosine':
        # Calculate the cosine distance matrix
        distance_matrix = pairwise_distances(data, metric='cosine')
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed')
        cluster_labels = clusterer.fit_predict(distance_matrix)
    else:
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)
        cluster_labels = clusterer.fit_predict(data)
    
    return cluster_labels, clusterer

# Experimenting with different distance metrics
distance_metrics = ['euclidean', 'manhattan', 'cosine']
results = {}

for metric in distance_metrics:
    print(f"\nClustering with distance metric: {metric}")
    
    # Perform clustering
    cluster_labels, clusterer = perform_hdbscan_clustering(reduced_images, min_cluster_size=5, min_samples=2, metric=metric)

    # Extract and Display Cluster Labels
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print("Number of points in each cluster:")
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} points")
        
    # Evaluate and Print Silhouette Score
    if len(set(cluster_labels)) > 1:
        score = silhouette_score(reduced_images, cluster_labels)
        print(f'Silhouette Score: {score}')
    else:
        print('Not enough clusters to calculate silhouette score.')

    # Visualization of Clusters
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_images[:, 0], reduced_images[:, 1], c=cluster_labels, cmap='viridis', marker='o', s=50)
    plt.title(f'HDBSCAN Clustering Results - Metric: {metric}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster Label')
    plt.grid()
    plt.savefig(f'hdbscan_clusters_noisy_{metric}.png')  
    plt.close()

    # Noise Analysis
    noise_mask = cluster_labels == -1
    num_noise_points = np.sum(noise_mask)
    print(f'Number of noise points identified: {num_noise_points}')
    
    # Visualizing Noise Points
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_images[noise_mask, 0], reduced_images[noise_mask, 1], color='red', alpha=0.8)
    plt.title('Distribution of Noise Points')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.savefig(f'noise_points_distribution_{metric}.png')  
    plt.close()

    # Silhouette Plot
    if len(set(cluster_labels)) > 1:
        sample_silhouette_values = silhouette_samples(reduced_images, cluster_labels)
        plt.figure(figsize=(10, 8))
        y_lower = 10  # Starting position
        for i in unique_labels:
            if i == -1:
                continue  # Skip noise points
            size_cluster_i = np.sum(cluster_labels == i)
            sort_values = np.sort(sample_silhouette_values[cluster_labels == i])
            plt.fill_betweenx(np.arange(y_lower, y_lower + size_cluster_i),
                              0, sort_values, alpha=0.7, label=f'Cluster {i}')
            y_lower += size_cluster_i

        average_silhouette = np.mean(sample_silhouette_values)
        plt.axvline(x=average_silhouette, color='red', linestyle='--')
        plt.title('Silhouette Plot for Clustering')
        plt.xlabel('Silhouette Coefficient Values')
        plt.ylabel('Cluster')
        plt.legend(loc='best')
        plt.savefig(f'silhouette_plot_{metric}.png')  
        plt.close()

    # Calculate the cluster centers (mean)
    cluster_centers = {}
    for label in unique_labels:
        if label != -1:  # Ignore noise
            cluster_points = reduced_images[cluster_labels == label]
            center = np.mean(cluster_points, axis=0)
            cluster_centers[label] = center

# Step 7: Real-Life Applications

# Function to preprocess a new image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((100, 100))  # Resize to match training images size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return img_array

# Example of adding a new image (the path should point to your new image)
new_image_path = '/home/fola/Downloads/test.jpg'  # Change this to your new image's path
new_image = preprocess_image(new_image_path)

# Flatten and standardize the new image
new_image_flat = new_image.flatten().reshape(1, -1)
new_image_standardized = scaler.transform(new_image_flat)  # Use the previously fitted scaler

# Reduce dimensionality of the new image using the PCA fitted on the original data
new_image_reduced = pca.transform(new_image_standardized)

# Classify the new image based on its distance to cluster centers
if len(cluster_centers) > 0:  # Ensure that clusters exist
    distances = []
    for label, center in cluster_centers.items():
        distance = np.linalg.norm(new_image_reduced - center)
        distances.append((label, distance))
    
    nearest_cluster_label, nearest_distance = min(distances, key=lambda x: x[1])

    # Set a threshold to classify as noise or assign to a cluster
    threshold_distance = 0.5  # Modify based on your dataset and requirements
    if nearest_distance < threshold_distance:
        print(f'New image assigned to cluster: {nearest_cluster_label}')
    else:
        print('New image classified as noise (-1)')
else:
    print("No clusters found.")

# 2. Representative Images for Clusters
# Find the most representative image for each cluster
representative_images = {}
for label, center in cluster_centers.items():
    distances = np.linalg.norm(reduced_images[cluster_labels == label] - center, axis=1)
    closest_index = np.argmin(distances)
    representative_images[label] = {
        'image_index': np.where(cluster_labels == label)[0][closest_index],
        'center': center
    }

# Display representative images
for label, info in representative_images.items():
    image_index = info['image_index']
    rep_image = images[image_index]  # Fetch the original image using the index
    plt.imshow(rep_image)
    plt.title(f'Representative Image for Cluster {label}')
    plt.axis('off')
    plt.savefig(f'representative_image_cluster_{label}.png')  
    plt.close()
    print(f'Saved representative image for cluster {label} at index {image_index}')
