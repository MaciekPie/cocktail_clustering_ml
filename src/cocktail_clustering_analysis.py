import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def load_data(file_path):
    """Load the dataset from the provided file path."""
    cocktail_data = pd.read_json(file_path)
    cocktail_data["tags"] = cocktail_data["tags"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    cocktail_data["num_ingredients"] = cocktail_data["ingredients"].apply(len)
    return cocktail_data


def perform_eda(data):
    """Perform exploratory data analysis and print key insights."""
    print("Dataset Info:")
    data.info()
    print("\nMissing Values:\n", data.isnull().sum())
    print("\nTag Counts:\n", data.explode("tags")["tags"].value_counts())
    print("\nCategory Counts:\n", data["category"].value_counts())
    print("\nIngredients Count Summary:\n", data["num_ingredients"].describe())


def preprocess_data(data):
    """Preprocess the dataset: one-hot encoding and normalization."""
    # One-hot encode categorical features
    category_encoded = pd.get_dummies(data["category"], prefix="category")
    glass_encoded = pd.get_dummies(data["glass"], prefix="glass")
    tags_encoded = data.explode("tags")
    tags_encoded = (
        pd.get_dummies(tags_encoded["tags"], prefix="tag").groupby(level=0).sum()
    )

    # Concatenate encoded columns and drop unnecessary ones
    data_encoded = pd.concat(
        [data, category_encoded, glass_encoded, tags_encoded], axis=1
    )
    numeric_data = data_encoded.drop(
        [
            "id",
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

    # Normalize numeric data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(numeric_data)
    return pd.DataFrame(normalized_data, columns=numeric_data.columns), data_encoded


def apply_kmeans(data, n_clusters=5):
    """Apply K-Means clustering to the dataset and return labels and Silhouette Score."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    sil_score = silhouette_score(data, labels)
    return labels, sil_score


def reduce_dimensions(data, n_components=2):
    """Reduce the dimensionality of the dataset for visualization."""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)
