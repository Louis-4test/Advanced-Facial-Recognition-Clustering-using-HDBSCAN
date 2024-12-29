import matplotlib
matplotlib.use('Agg')  
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
from sklearn.metrics import silhouette_score
import warnings
from hdbscan import HDBSCAN 

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

# Load and explore dataset
max_images_to_load = 100
images = []
labels = []

for img, label in lazy_load_images(base_directory, max_images=max_images_to_load):
    images.append(img)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

num_images = len(images)
image_shape = images[0].shape if num_images > 0 else None

print(f'Total images loaded: {num_images}')
print(f'Image shape: {image_shape}')
print(f'Unique labels: {np.unique(labels)}')

# Save the loaded images to files
for i in range(min(15, num_images)):
    plt.imsave(f'image_{i}.png', images[i])

print("Images saved successfully.")


# Load and explore dataset
max_images_to_load = 100
images = []
labels = []

for img, label in lazy_load_images(base_directory, max_images=max_images_to_load):
    images.append(img)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Flatten the images
num_images, height, width, channels = images.shape
flattened_images = images.reshape(num_images, height * width * channels)
print(f'Shape of flattened images: {flattened_images.shape}')

# Standardize the dataset
scaler = StandardScaler()
standardized_images = scaler.fit_transform(flattened_images)
print(f'Shape of standardized images: {standardized_images.shape}')

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

 # Reshape the images for PCA
num_samples, height, width, channels = images.shape
reshaped_images = images.reshape(num_samples, height * width * channels)  # Flatten each image

#PCA Reduction. Decide on the number of components to retain (e.g., 90% variance)
pca = PCA(n_components=0.90)  
reduced_images = pca.fit_transform(reshaped_images)

# Print relevant information
print(f"Number of components chosen to retain 90% variance: {pca.n_components_}")
print(f"Shape of reduced images: {reduced_images.shape}")

# Visualize PCA Reduced Data and Create a 2D visualization of the first two principal components
plt.scatter(reduced_images[:, 0], reduced_images[:, 1], alpha=0.5)
plt.title('PCA Reduced Data Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.savefig('pca_reduced_data.png')  
plt.close()

# Apply HDBSCAN Clustering with altered parameters. Scaling the reshaped images
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
reshaped_images_scaled = scaler.fit_transform(reshaped_images)

# Configure HDBSCAN parameters with adjustments
min_cluster_size = 2 
min_samples = 1 
distance_metric = 'manhattan'  

# Initialize and fit HDBSCAN
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=distance_metric)
cluster_labels = hdbscan_clusterer.fit_predict(reshaped_images_scaled)

#Extract and Display Cluster Labels
print("Cluster labels assigned to each data point:")
print(cluster_labels)

# Count the number of points in each cluster excluding noise (-1)
unique_labels, counts = np.unique(cluster_labels, return_counts=True)
print("Number of points in each cluster:")
for label, count in zip(unique_labels, counts):
    print(f"Cluster {label}: {count} points")

if len(set(cluster_labels)) > 1:
    score = silhouette_score(reduced_images, cluster_labels)
    print(f'Silhouette Score: {score}')
else:
    print('Not enough clusters to calculate silhouette score.')

# Optionally visualize the cluster results
plt.figure(figsize=(10, 8))
plt.scatter(reduced_images[:, 0], reduced_images[:, 1], c=cluster_labels, cmap='viridis', marker='o', s=50)
plt.title('HDBSCAN Clustering Results with Silhouette Score')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.grid()
plt.savefig('hdbscan_clusters_final_with_silhouette.png')  
plt.close()



