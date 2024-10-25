import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the cocktail dataset from the provided file path
file_path = "Uczenie/Solvro/Machine_Learning/cocktail_dataset.json"
cocktail_data = pd.read_json(file_path)

# Ensure missing values are handled in the 'tags' column
cocktail_data["tags"] = cocktail_data["tags"].apply(
    lambda x: x if isinstance(x, list) else []
)

# Add num_ingredients column before encoding
cocktail_data["num_ingredients"] = cocktail_data["ingredients"].apply(len)

# One-hot encode the category, glass, and tags columns
category_encoded = pd.get_dummies(cocktail_data["category"], prefix="category")
glass_encoded = pd.get_dummies(cocktail_data["glass"], prefix="glass")
tags_encoded = cocktail_data.explode("tags")
tags_encoded = pd.get_dummies(tags_encoded["tags"], prefix="tag").groupby(level=0).sum()

# Concatenate all the one-hot encoded columns with the original dataframe
cocktail_data_encoded = pd.concat(
    [cocktail_data, category_encoded, glass_encoded, tags_encoded], axis=1
)

# Drop non-numeric columns but KEEP 'num_ingredients'
numeric_data = cocktail_data_encoded.drop(
    [
        "id",  # Dropping id since it's not useful for clustering
        "name",
        "category",
        "glass",
        "tags",
        "createdAt",
        "updatedAt",
        "instructions",
        "imageUrl",
        "ingredients",
    ],
    axis=1,
)

# Normalize the numerical columns
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(numeric_data)

# Convert back to DataFrame for easy handling
normalized_df = pd.DataFrame(normalized_data, columns=numeric_data.columns)

# Apply K-Means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(normalized_df)

# Add the cluster labels to the dataframe
cocktail_data_encoded["cluster"] = kmeans.labels_

# Evaluate the clustering with Silhouette Score
sil_score_kmeans = silhouette_score(normalized_df, kmeans.labels_)
print(f"Silhouette Score (K-Means): {sil_score_kmeans}")

# PCA Visualization (K-Means)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_df)

# Plot the K-Means clusters
plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans.labels_, cmap="viridis")
plt.title("Cocktail Clusters (PCA) - K-Means")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Cluster")
plt.show()

# Analyze clusters based on key numeric features
# Prepare DataFrame with numeric columns only
numeric_columns_for_analysis = ["cluster", "num_ingredients", "category_Cocktail"]

# Optionally, include specific tag columns or all tag columns
tag_columns = [col for col in cocktail_data_encoded.columns if col.startswith("tag_")]
numeric_columns_for_analysis.extend(tag_columns)

# Group by cluster and calculate mean
cluster_analysis = cocktail_data_encoded[numeric_columns_for_analysis]
cluster_means = cluster_analysis.groupby("cluster").mean()

print("Cluster Means:")
print(cluster_means)
