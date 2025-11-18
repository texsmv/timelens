# timelens/clustering.py
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from typing import Dict, List, Tuple, Union


def perform_kmeans(data: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> Dict[str, Union[List[int], Dict[str, str]]]:
    """
    Perform K-means clustering on 2D projection data.
    
    Args:
        data: 2D numpy array of shape (n_points, 2) containing projection coordinates
        n_clusters: Number of clusters to create
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing:
        - 'values': List of cluster labels (integers)
        - 'labels': Dictionary mapping cluster IDs to human-readable names
    """
    if data.shape[0] < n_clusters:
        raise ValueError(f"Number of data points ({data.shape[0]}) must be >= n_clusters ({n_clusters})")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    
    # Create labels dictionary
    labels_dict = {}
    for i in range(n_clusters):
        labels_dict[str(i)] = f"Cluster {i + 1}"
    
    return {
        'values': cluster_labels.tolist(),
        'labels': labels_dict
    }


def perform_dbscan(data: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Union[List[int], Dict[str, str]]]:
    """
    Perform DBSCAN clustering on 2D projection data.
    
    Args:
        data: 2D numpy array of shape (n_points, 2) containing projection coordinates
        eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_samples: Number of samples in a neighborhood for a point to be considered as a core point
        
    Returns:
        Dictionary containing:
        - 'values': List of cluster labels (integers, -1 for outliers)
        - 'labels': Dictionary mapping cluster IDs to human-readable names
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(data)
    
    # Get unique labels and create labels dictionary
    unique_labels = np.unique(cluster_labels)
    labels_dict = {}
    
    for label in unique_labels:
        if label == -1:
            labels_dict[str(label)] = "Outliers"
        else:
            labels_dict[str(label)] = f"Cluster {label + 1}"
    
    return {
        'values': cluster_labels.tolist(),
        'labels': labels_dict
    }


def get_clustering_result(
    data: List[List[float]], 
    algorithm: str, 
    **kwargs
) -> Dict[str, Union[List[int], Dict[str, str]]]:
    """
    Main function to perform clustering based on the specified algorithm.
    
    Args:
        data: List of [x, y] coordinates for each point
        algorithm: Either 'kmeans' or 'dbscan'
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Dictionary containing cluster labels and human-readable names
        
    Raises:
        ValueError: If algorithm is not supported or parameters are invalid
    """
    # Convert to numpy array
    data_array = np.array(data)
    
    if data_array.shape[1] != 2:
        raise ValueError(f"Data must have 2 dimensions, got {data_array.shape[1]}")
    
    if data_array.shape[0] < 2:
        raise ValueError("At least 2 data points are required for clustering")
    
    if algorithm.lower() == 'kmeans':
        n_clusters = kwargs.get('n_clusters', 3)
        random_state = kwargs.get('random_state', 42)
        
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        
        return perform_kmeans(data_array, n_clusters=n_clusters, random_state=random_state)
    
    elif algorithm.lower() == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        
        return perform_dbscan(data_array, eps=eps, min_samples=min_samples)
    
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'kmeans' or 'dbscan'")


def estimate_dbscan_eps(data: List[List[float]], k: int = 4) -> float:
    """
    Estimate a good eps value for DBSCAN using the k-distance graph method.
    
    Args:
        data: List of [x, y] coordinates for each point
        k: Number of nearest neighbors to consider
        
    Returns:
        Suggested eps value
    """
    from sklearn.neighbors import NearestNeighbors
    
    data_array = np.array(data)
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data_array)
    distances, indices = neighbors_fit.kneighbors(data_array)
    
    # Sort the distances to the k-th nearest neighbor
    distances = np.sort(distances[:, k-1], axis=0)
    
    # Use the knee point as eps estimate (simplified approach)
    # In practice, you might want to use more sophisticated knee detection
    knee_idx = len(distances) // 2
    return distances[knee_idx]