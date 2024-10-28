# Cocktail clustering

## Overview

Exploratory data analysis and clustering of cocktail data using Machine Learning for the Solvro student research group.

## Project Structure

- **/data**: Contains the dataset in JSON format.
- **/src**: Python script for loading, processing, and clustering the data.
- **/notebook**: Jupyter Notebook used for EDA and visualization.
- **requirements.txt**: Lists the necessary libraries for the project.
- **LICENSE**: MIT License file.

## Project Details

1. Exploratory Data Analysis (EDA)

   The EDA phase examines the distribution of cocktail categories, common ingredients, and tags. Key steps include:

   - Identifying missing values and common tags.
   - Analyzing the distribution of ingredients and categories.

2. Data Preprocessing

   The data preprocessing involves:

   - One-hot encoding of categorical features (e.g., category, glass type, and tags).
   - Normalizing numerical data for clustering.

3. Clustering

   We apply K-Means clustering with 5 clusters based on cocktail ingredients and characteristics:

   - Clustering Metric: We calculate the Silhouette Score to evaluate cluster quality.
   - Dimensionality Reduction: PCA is used for a 2D visualization of clusters.

4. Evaluation

   - Silhouette Score: Provides a quantitative measure of clustering quality.
   - PCA Visualization: Aids in visualizing the distribution of cocktails across clusters.

5. Exploratory Data Analysis (EDA)

   The EDA phase examines the distribution of cocktail categories, common ingredients, and tags. Key steps include:

   - Identifying missing values and common tags.
   - Analyzing the distribution of ingredients and categories.

## Installation

1. **Clone the repository** to your local environment:

   ```bash
   git clone https://github.com/MaciekPie/cocktail_clustering_ml
   ```

2. Create new virtual environment:

   ```bash
   conda create --name {env_name}
   ```

   Alternatively use any other virtual environment of your choice.

3. Activate the environment:

   ```bash
   conda activate {env_name}
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the Jupyter Notebook:
   Navigate to the notebook directory and launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Open cocktail_clustering_notebook.ipynb and run each cell to perform EDA, clustering, and visualization.

6. Start coding!

## Key Insights

- The dataset groups into five clusters based on cocktail ingredients and categories.

- Moderate separation of clusters, as indicated by the Silhouette Score, suggests meaningful clustering.
