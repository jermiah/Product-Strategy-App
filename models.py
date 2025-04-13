import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE



def kmeans_euclidean_eval(scaled_features, k_values, random_state=42):
    """
    Evaluate K-Means (Euclidean) for multiple k values.
    
    Returns:
      inertia_scores: list of inertia (WCSS) for each k
      silhouette_scores: list of silhouette scores (Euclidean) for each k
    """
    inertia_scores = []
    silhouette_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(scaled_features)
        
        # Inertia (WCSS)
        inertia = kmeans.inertia_
        inertia_scores.append(inertia)
        
        # Silhouette Score (Euclidean)
        sil_score = silhouette_score(scaled_features, labels, metric='euclidean')
        silhouette_scores.append(sil_score)
    
    return inertia_scores, silhouette_scores



def kmedoids_manhattan_eval(scaled_features, k_values, random_state=42):
    """
    Evaluate K-Medoids (Manhattan) for multiple k values.
    
    Returns:
      medoid_costs: list of sum of distances to medoids for each k
      silhouette_scores: list of silhouette scores (Manhattan) for each k
    """
    medoid_costs = []
    silhouette_scores = []
    
    for k in k_values:
        kmedoids = KMedoids(n_clusters=k, metric='manhattan', random_state=random_state)
        labels = kmedoids.fit_predict(scaled_features)
        
        # Medoid Cost: Sum of distances from each point to its medoid
        total_cost = np.sum(np.min(kmedoids.transform(scaled_features), axis=1))
        medoid_costs.append(total_cost)
        
        # Silhouette Score (Manhattan)
        sil_score = silhouette_score(scaled_features, labels, metric="manhattan")
        silhouette_scores.append(sil_score)
    
    return medoid_costs, silhouette_scores




def kmedoids_cosine_eval(scaled_features, k_values, random_state=42):
    """
    Evaluate K-Medoids (Cosine Distance) for multiple k values.
    
    Returns:
      medoid_costs: list of sum of distances to medoids for each k
      silhouette_scores: list of silhouette scores (precomputed) for each k
    """
    # Precompute Cosine Distance Matrix
    cosine_dist_matrix = cosine_distances(scaled_features)
    
    medoid_costs = []
    silhouette_scores = []
    
    for k in k_values:
        kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=random_state)
        labels = kmedoids.fit_predict(cosine_dist_matrix)
        
        # Medoid Cost: Sum of distances from each point to its medoid
        total_cost = np.sum(np.min(kmedoids.transform(cosine_dist_matrix), axis=1))
        medoid_costs.append(total_cost)
        
        # Silhouette Score (Cosine Distance)
        sil_score = silhouette_score(cosine_dist_matrix, labels, metric="precomputed")
        silhouette_scores.append(sil_score)
    
    return medoid_costs, silhouette_scores



def spectral_cosine_eval(scaled_features, k_values, random_state=42):
    """
    Evaluate Spectral Clustering (Cosine Similarity) for multiple k values.
    Uses:
      - Clamped negative values to 0
      - Diagonal set to 1.0
      - Converts similarity -> distance for silhouette
    Returns:
      silhouette_scores: list of silhouette scores for each k
    """
    # Step 1: Compute Cosine Similarity
    cos_sim_matrix = cosine_similarity(scaled_features)
    
    # Step 2: Fix negative values, fill diagonal
    cos_sim_matrix = np.clip(cos_sim_matrix, 0, 1)
    np.fill_diagonal(cos_sim_matrix, 1.0)
    cos_sim_matrix = np.nan_to_num(cos_sim_matrix, nan=0.0)
    
    # Step 3: Convert to distance for silhouette scoring
    cos_dist_matrix = 1 - cos_sim_matrix
    
    silhouette_scores = []
    
    for k in k_values:
        spectral = SpectralClustering(
            n_clusters=k,
            affinity='precomputed',
            random_state=random_state,
            eigen_solver='arpack',
            assign_labels='discretize'
        )
        labels = spectral.fit_predict(cos_sim_matrix)
        
        # Compute Silhouette Score using distance matrix
        sil_score = silhouette_score(cos_dist_matrix, labels, metric="precomputed")
        silhouette_scores.append(sil_score)
    
    return silhouette_scores


def plot_elbow_silhoutte(k_values, medoid_costs, silhouette_scores):
    """
    Plots the K-Medoids cost function and silhouette score for different cluster counts.

    Parameters:
        k_values (list or array-like): The different k (cluster count) values.
        medoid_costs (list or array-like): The medoid costs corresponding to each k.
        silhouette_scores (list or array-like): The silhouette scores corresponding to each k.
    """
    plt.figure(figsize=(10, 4))
    
    # Medoid Cost Plot (Alternative to Elbow Method)
    plt.subplot(1, 2, 1)
    plt.plot(k_values, medoid_costs, marker='o', linestyle='-')
    plt.xticks(k_values)  # Force integer tick marks
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Medoid Cost (Sum of Distances to Medoids)")
    plt.title("K-Medoids Cost Function")
    
    # Silhouette Score Plot
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='orange')
    plt.xticks(k_values)  # Force integer tick marks
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score by k")
    
    plt.tight_layout()
    plt.show()

def run_cluster_evaluation(scaled_features, k_values, random_state=42):
    """
    Run evaluations for various clustering methods and return their results.

    Parameters:
        scaled_features (array-like): Scaled feature matrix (samples x features).
        k_values (iterable): A range/list of k values to evaluate.
        random_state (int): Random state for reproducibility.

    Returns:
        dict: A dictionary containing results for each clustering method.
    """
    # Evaluate K-Means (Euclidean)
    inertia_scores, silhouette_scores_kmeans = kmeans_euclidean_eval(scaled_features, k_values, random_state)
    # print("K-Means (Euclidean) Evaluation:")
    # print("Inertia Scores:", inertia_scores)
    # print("Silhouette Scores:", silhouette_scores_kmeans)
    
    # Evaluate K-Medoids (Manhattan)
    medoid_costs, silhouette_scores_kmedoids = kmedoids_manhattan_eval(scaled_features, k_values, random_state)
    # print("\nK-Medoids (Manhattan) Evaluation:")
    # print("Medoid Costs:", medoid_costs)
    # print("Silhouette Scores:", silhouette_scores_kmedoids)
    
    # Evaluate K-Medoids (Cosine)
    medoid_costs_cosine, silhouette_scores_kmedoids_cosine = kmedoids_cosine_eval(scaled_features, k_values, random_state)
    # print("\nK-Medoids (Cosine) Evaluation:")
    # print("Medoid Costs:", medoid_costs_cosine)
    # print("Silhouette Scores:", silhouette_scores_kmedoids_cosine)
    
    # Evaluate Spectral Clustering (Cosine Similarity)
    silhouette_scores_spectral = spectral_cosine_eval(scaled_features, k_values, random_state)
    # print("\nSpectral Clustering (Cosine Similarity) Evaluation:")
    # print("Silhouette Scores:", silhouette_scores_spectral)
    
    # Return all evaluation results in a dictionary
    return {
        'kmeans': {'inertia_scores': inertia_scores, 'silhouette_scores': silhouette_scores_kmeans},
        'kmedoids_manhattan': {'medoid_costs': medoid_costs, 'silhouette_scores': silhouette_scores_kmedoids},
        'kmedoids_cosine': {'medoid_costs': medoid_costs_cosine, 'silhouette_scores': silhouette_scores_kmedoids_cosine},
        'spectral': {'silhouette_scores': silhouette_scores_spectral}
    }


def create_comparison_df(results, k_values):
    """
    Create two DataFrames:
    1. Comparison DataFrame with clustering evaluation metrics for each method.
    2. Averaged Metrics DataFrame with each metric's average value across all k-values.
    
    Parameters:
        results (dict): Dictionary returned by run_cluster_evaluation.
        k_values (iterable): The range/list of k values evaluated.
        
    Returns:
        tuple: (comparison_df, average_df)
    """
    comparison_df = pd.DataFrame({'k': k_values})
    
    # For K-Means (Euclidean)
    comparison_df['kmeans_inertia'] = results['kmeans']['inertia_scores']
    comparison_df['kmeans_silhouette'] = results['kmeans']['silhouette_scores']
    
    # For K-Medoids (Manhattan)
    comparison_df['kmedoids_manhattan_cost'] = results['kmedoids_manhattan']['medoid_costs']
    comparison_df['kmedoids_manhattan_silhouette'] = results['kmedoids_manhattan']['silhouette_scores']
    
    # For K-Medoids (Cosine)
    comparison_df['kmedoids_cosine_cost'] = results['kmedoids_cosine']['medoid_costs']
    comparison_df['kmedoids_cosine_silhouette'] = results['kmedoids_cosine']['silhouette_scores']
    
    # For Spectral Clustering (Cosine Similarity)
    comparison_df['spectral_silhouette'] = results['spectral']['silhouette_scores']
    
    # Compute average silhouette score across all methods
    silhouette_cols = ['kmeans_silhouette', 'kmedoids_manhattan_silhouette', 'kmedoids_cosine_silhouette', 'spectral_silhouette']
   
    
    # Compute average cost (excluding silhouette scores as they are different metrics)
    cost_cols = ['kmedoids_manhattan_cost', 'kmedoids_cosine_cost']
 

    # Create a second DataFrame that converts columns into rows and calculates overall averages
    average_df = comparison_df.drop(columns=['k']).mean().reset_index()
    average_df.columns = ['Metric', 'Average Value']
    
    return comparison_df, average_df


def compute_cluster_and_silhouette(model, scaled_features, transformation="none", metric=None, title=None):

       
    """
    Plot a horizontal silhouette plot for a clustering model.

    Parameters:
      model: clustering model instance with a fit_predict method.
             For transformations "cosine" or "spectral", the model should be set up to accept a precomputed distance matrix.
      scaled_features: raw feature array (already scaled).
      transformation: one of "none", "cosine", or "spectral".
          - "none": Use the scaled_features directly, with the provided metric (default 'euclidean').
          - "cosine": Precompute the cosine distance matrix.
          - "spectral": Precompute the cosine similarity matrix, clamp values, then convert to distance (1 - sim).
      metric: metric string to be used in silhouette computations. If transformation != "none", this is forced to 'precomputed'.
      title: Title for the plot (if None, defaults to model's class name).
      
    Returns:
      cluster_labels: array of cluster labels.
      silhouette_avg: average silhouette score.
    """
    # Transform data based on the transformation flag
    if transformation == "cosine":
        # Precompute cosine distance matrix
        data_transformed = cosine_distances(scaled_features)
        metric_used = "precomputed"
    elif transformation == "spectral":
        # Compute cosine similarity, then transform to distance
        cos_sim_matrix = cosine_similarity(scaled_features)
        # Clamp negative values to 0 and ensure diagonal is 1
        cos_sim_matrix = np.clip(cos_sim_matrix, 0, 1)
        np.fill_diagonal(cos_sim_matrix, 1.0)
        cos_sim_matrix = np.nan_to_num(cos_sim_matrix, nan=0.0)
        data_transformed = 1 - cos_sim_matrix
        metric_used = "precomputed"
    else:
        # Use the raw scaled features
        data_transformed = scaled_features
        metric_used = metric if metric is not None else "euclidean"

    # Fit the clustering model and predict labels
    cluster_labels = model.fit_predict(data_transformed)
    
    # Compute silhouette values for each sample
    sil_vals = silhouette_samples(data_transformed, cluster_labels, metric=metric_used)
    silhouette_avg = silhouette_score(data_transformed, cluster_labels, metric=metric_used)
    print(f"Average Silhouette Score ({model.__class__.__name__}): {silhouette_avg:.4f}")

    # Create silhouette plot
    fig, ax = plt.subplots(figsize=(10, 7))
    n_clusters = np.unique(cluster_labels).shape[0]
    y_lower = 0
    yticks = []

    for i, cluster in enumerate(np.unique(cluster_labels)):
        # Collect and sort the silhouette scores for samples in this cluster
        cluster_sil_vals = sil_vals[cluster_labels == cluster]
        cluster_sil_vals.sort()
        cluster_size = len(cluster_sil_vals)
        y_upper = y_lower + cluster_size

        # Choose a color from the colormap
        color = cm.jet(float(i) / n_clusters)
        ax.barh(np.arange(y_lower, y_upper),
                cluster_sil_vals,
                height=1.0,
                edgecolor='none',
                color=color)

        # Append midpoint for y-axis labeling
        yticks.append((y_lower + y_upper) / 2)
        y_lower = y_upper + 10  # gap between clusters

    ax.axvline(silhouette_avg, color='red', linestyle='--', label="Avg Silhouette Score")
    ax.set_yticks(yticks)
    # Label clusters starting at 1 for readability
    ax.set_yticklabels(np.unique(cluster_labels) + 1)
    ax.set_ylabel('Cluster')
    ax.set_xlabel('Silhouette Coefficient')
    if title is None:
        title = f"Silhouette Plot ({model.__class__.__name__})"
    ax.set_title(title)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

    return cluster_labels, silhouette_avg


def plot_numeric_histograms(df, numeric_cols, bins=100):
    """
    Plots histograms with KDE overlays for all numerical columns in the given DataFrame.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
        numeric_cols (list): List of column names of numerical features.
        bins (int): Number of bins for the histograms. Default is 100.
    """
    # Calculate the number of rows for subplots (3 plots per row)
    rows = (len(numeric_cols) - 1) // 3 + 1
    
    # Set the overall figure size
    plt.figure(figsize=(15, 3 * rows + 3))
    
    # Loop through each numeric column and plot a histogram with a KDE overlay
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(rows, 3, i)
        sns.histplot(df[col], kde=True, bins=bins)
        plt.title(col)
        plt.xlabel("")
        plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()



def plot_numeric_boxplots(df,numeric_cols):
    """
    Plots box plots for all numeric columns in the given DataFrame.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to plot.
    """
   
    
    # Calculate the number of rows for subplots (3 plots per row)
    rows = ((len(numeric_cols) - 1) // 3) + 1
    
    # Set the overall figure size; adjust the height based on the number of numeric features
    plt.figure(figsize=(15, 3 * rows))
    
    # Loop through each numeric column and plot a box plot
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(rows, 3, i)
        sns.boxplot(y=df[col])
        plt.title(col)
        plt.xlabel("")
        plt.ylabel("")
    
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, figsize=(10, 6), annot=True, cmap="coolwarm", fmt=".2f", title="Correlation Matrix"):
    """
    Plots a heatmap for the correlation matrix of all numeric columns in the given DataFrame.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame to plot.
        figsize (tuple): Size of the figure. Default is (10, 6).
        annot (bool): Whether to annotate the heatmap with correlation values. Default is True.
        cmap (str): Colormap to use. Default is "coolwarm".
        fmt (str): String formatting code for annotations. Default is ".2f".
        title (str): Title of the plot. Default is "Correlation Matrix".
    """
    plt.figure(figsize=figsize)
    # Compute the correlation matrix for numeric columns
    corr_matrix = df.select_dtypes(include=['number']).corr()
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=fmt)
    plt.title(title)
    plt.show()


def plot_pca_clusters(scaled_features, cluster_labels, title="Cluster Visualization using PCA"):
    """
    Apply PCA and visualize the clusters in 2D space.
    
    Parameters:
        scaled_features (array-like): The scaled feature matrix.
        cluster_labels (array-like): The cluster labels from a clustering model.
        title (str): Title for the PCA plot.
    
    Returns:
        None
    """
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.colorbar(scatter, label="Cluster")
    plt.show()



def plot_product_clusters(df, cluster_labels, method="tsne", title="Product Clusters in 2D", perplexity=40, show_labels=True):
    """
    Visualizes product clusters in 2D space using t-SNE or PCA.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame containing product data.
        cluster_labels (array-like): Cluster labels assigned to each product.
        method (str): Dimensionality reduction method ("tsne" or "pca").
        title (str): Title for the plot.
        perplexity (int): The perplexity parameter for t-SNE (default=40).
        show_labels (bool): Whether to display product names on data points (default=True).
    
    Returns:
        None
    """
    # Reduce dimensions using t-SNE or PCA
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    reduced_features = reducer.fit_transform(df)

    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        x=reduced_features[:, 0], 
        y=reduced_features[:, 1], 
        hue=cluster_labels,  # Always keep clusters
        palette="Set1", 
        alpha=0.8, 
        legend="full"
    )

    # Annotate product names only if show_labels=True
    if show_labels:
        num_labels = min(150, len(df))  # Show at most 100 labels
        indices = np.random.choice(len(df), num_labels, replace=False)  # Randomly select products for labeling
        
        for i in indices:
            plt.text(
                reduced_features[i, 0], reduced_features[i, 1], 
                str(df.index[i]),  # Assuming product names are in index
                fontsize=8, color='black', alpha=0.75
            )

    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.title(title)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.show()


def improved_tsne_plot(df, cluster_labels, perplexity=40,  title="Improved Product Clusters in 2D"):
    """
    Creates an improved t-SNE visualization with better readability.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame containing product features.
        cluster_labels (array-like): Cluster labels assigned to each product.
        title (str): Title for the plot.
    
    Returns:
        None
    """
    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_features = tsne.fit_transform(df)
    
    # Create scatter plot
    plt.figure(figsize=(14, 9))
    scatter = sns.scatterplot(
        x=reduced_features[:, 0], 
        y=reduced_features[:, 1], 
        hue=cluster_labels, 
        palette="Set1", 
        alpha=0.7, 
        s=80
    )

    # Selectively annotate product names
    num_labels = min(150, len(df))  # Show at most 150 labels
    indices = np.random.choice(len(df), num_labels, replace=False)  # Randomly choose products for labeling

    for i in indices:
        plt.text(
            reduced_features[i, 0], reduced_features[i, 1], 
            df.index[i],  # Assuming product names are in index
            fontsize=9, color='black', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle="round,pad=0.3")
        )

    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)
    plt.title(title, fontsize=15, fontweight="bold")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()



# Clear plotting function with data labels
def plot_cluster_metrics(df, base_metrics, title,segmentation_column):
    stats = ['mean', 'median', 'std']
    clusters = df[segmentation_column]
    num_clusters = len(clusters)
    bar_width = 0.18
    indices = np.arange(num_clusters)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for idx, stat in enumerate(stats):
        ax = axes[idx]

        for i, metric in enumerate(base_metrics):
            metric_name = f"{metric}_{stat}"
            values = df[metric_name]
            bars = ax.bar(indices + i * bar_width, values, bar_width, label=metric.replace('_', ' '))

            # Adding data labels clearly
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Clusters', fontsize=14)
        ax.set_ylabel(f'{stat.capitalize()} Values', fontsize=14)
        ax.set_title(f'{stat.capitalize()} Comparison', fontsize=16)
        ax.set_xticks(indices + bar_width * (len(base_metrics) - 1) / 2)
        ax.set_xticklabels(clusters, fontsize=12)
        ax.legend(title='Metrics')

    fig.suptitle(title, fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

