import numpy as np
import os
from typing import Dict, List, Optional, Any

from timelens.utils import project_mts


class MultivariateTimeSeries:
    """
    Represents a multivariate time series (MTS) object.

    Attributes:
        name (str): Name of the MTS.
        mts (np.ndarray): The MTS data, expected shape (N, T, D).
                          N: number of series, T: time points, D: dimensions.
        dimensions (Optional[List[str]]): Names of the dimensions (features).
        coords (Optional[Dict[str, np.ndarray]]): Dictionary of projections/coordinates,
                                                 e.g., {'pca': np.ndarray(N, 2)}.
        labels (Optional[Dict[str, np.ndarray]]): Dictionary of labels for the series,
                                                 e.g., {'cluster_ids': np.ndarray(N,)}.
        label_names (Optional[Dict[str, Dict[int, str]]]): Names for the labels,
                                                          e.g., {'cluster_ids': {0: 'Cluster A', 1: 'Cluster B'}}.
    """
    def __init__(
            self,
            name: str,
            mts: np.ndarray,
            dimensions: Optional[List[str]] = None,
            coords: Optional[Dict[str, np.ndarray]] = None,
            labels: Optional[Dict[str, np.ndarray]] = None,
            label_names: Optional[Dict[str, Dict[int, str]]] = None
    ):
        if not isinstance(name, str):
            raise TypeError("Name must be a string.")
        if not isinstance(mts, np.ndarray):
            raise TypeError("mts must be a NumPy array.")
        if mts.ndim != 3:
            raise ValueError("mts must be a 3D array of shape (N, T, D).")
        if dimensions is not None and not isinstance(dimensions, list):
            raise TypeError("dimensions must be a list of strings or None.")
        if coords is not None and not isinstance(coords, dict):
            raise TypeError("coords must be a dictionary or None.")
        if labels is not None and not isinstance(labels, dict):
            raise TypeError("labels must be a dictionary or None.")
        if label_names is not None and not isinstance(label_names, dict):
            raise TypeError("label_names must be a dictionary or None.")

        self.name = name
        self.mts = mts
        self.dimensions = dimensions if dimensions is not None else None
        self.coords = coords if coords is not None else None
        self.labels = labels if labels is not None else None
        self.label_names = label_names if label_names is not None else None

    def get_sampled_object(self, n: int):
        """
        Returns a new MultivariateTimeSeries object with only n samples from the original object.
        """
        # Randomly sample n indices
        indices = np.random.choice(self.mts.shape[0], n, replace=False)
        return MultivariateTimeSeries(
            name=self.name,
            mts=self.mts[indices],
            dimensions=self.dimensions,
            coords={k: v[indices] for k, v in self.coords.items()} if self.coords is not None else None,
            labels={k: v[indices] for k, v in self.labels.items()} if self.labels is not None else None,
            label_names=self.label_names
        )


class MTSStorage:
    """
    Class to store and load MultivariateTimeSeries objects to/from a file.

    Uses numpy.save and numpy.load for efficient storage.
    """
    def __init__(self, filename: str):
        """
        Initializes MTSStorage with a filename for saving/loading data.

        Args:
            filename (str): Base filename (without extension) for the storage file.
                           The '.npy' extension will be added automatically.
        """
        self.filename = filename
        self.objects: Dict[str, MultivariateTimeSeries] = {}  # Use type hint for objects

    def add_mts(
            self,
            name: str,
            mts: np.ndarray,
            dimensions: Optional[List[str]] = None,
            projection: Optional[np.ndarray] = None, # Shape N x 2
            labels: Optional[np.ndarray] = None, # Shape N
            label_names: Optional[Dict[int, str]] = None, # Dict e.g. {0: 'Group A', 1: 'Group B'}
            reducer = None,
    ):
        assert mts.ndim == 3, "MTS must be 3D array of shape (N, T, D)."
        assert dimensions is None or len(dimensions) == mts.shape[2], "Number of dimensions must match MTS shape."
        assert projection is None or projection.shape[0] == mts.shape[0], "Projection must have N rows."
        assert labels is None or labels.shape[0] == mts.shape[0], "Labels must have N rows."
        unique_labels = np.unique(labels) if labels is not None else None
        assert label_names is None or len(unique_labels) == len(label_names), "Number of label names must match unique labels."


        if projection is None:
            projection = project_mts(mts, reducer=reducer)

        mts_object = MultivariateTimeSeries(
            name=name,
            mts=mts,
            dimensions=dimensions,
            coords={'projection': projection} if projection is not None else None,
            labels={'labels': labels} if labels is not None else None,
            label_names={'label_names': label_names} if label_names is not None else None
        )
        self.objects[name] = mts_object


    def add_mts_multi(
            self,
            name: str,
            mts: np.ndarray,
            dimensions: Optional[List[str]] = None,
            coords: Optional[Dict[str, Any]] = None, # Allow generic dict for input, convert inside
            labels: Optional[Dict[str, Any]] = None, # Allow generic dict for input, convert inside
            label_names: Optional[Dict[str, Dict[int, str]]] = None
    ):
        """
        Adds a MultivariateTimeSeries object to the storage.

        Args:
            name (str): Name of the MTS.
            mts (np.ndarray): The MTS data (N, T, D).
            dimensions (Optional[List[str]]): List of dimension names.
            coords (Optional[Dict[str, Any]]): Dictionary of coordinates (projections).
                                                Values will be converted to np.ndarray.
            labels (Optional[Dict[str, Any]]): Dictionary of labels.
                                              Values will be converted to np.ndarray of integers.
            label_names (Optional[Dict[str, Dict[int, str]]]): Dictionary of label names.
        """

        # Process coords and labels to ensure they are numpy arrays
        processed_coords: Dict[str, np.ndarray] = {}
        if coords:
            for k, v in coords.items():
                processed_coords[k] = np.array(v)

        processed_labels: Dict[str, np.ndarray] = {}
        if labels:
            for k, v in labels.items():
                processed_labels[k] = np.array(v).astype(int)

        mts_object = MultivariateTimeSeries(
            name=name,
            mts=mts,
            dimensions=dimensions,
            coords=processed_coords,
            labels=processed_labels,
            label_names=label_names
        )
        self.objects[name] = mts_object

    def get_mts(self, name: str, max_windows: None) -> Optional[MultivariateTimeSeries]:
        """
            Returns the MTS object with the given name, or None if not found.
            If max_windows is provided, limits the number of windows to return in the [MultivariateTimeSeries]
        """
        if not name in self.objects:
            return None
        
        mts = self.objects[name]

        if max_windows is None:
            return mts
        
        return mts.get_sampled_object(max_windows)
        


    def save(self):
        """Saves the MTS objects to a .npy file using numpy.save."""
        objects_to_save = {name: vars(mts_obj) for name, mts_obj in self.objects.items()} # Save MTS attributes as dicts
        np.save(self.filename + '.npy', objects_to_save, allow_pickle=True)

    def load(self):
        """Loads MTS objects from a .npy file using numpy.load."""
        if os.path.exists(self.filename + '.npy'):
            loaded_objects_dict = np.load(self.filename + '.npy', allow_pickle=True).item() # Load as dict
            self.objects = {}
            for name, obj_dict in loaded_objects_dict.items():
                self.objects[name] = MultivariateTimeSeries(**obj_dict) # Reconstruct MTS objects

    def delete(self):
        """Deletes the storage file if it exists."""
        if os.path.exists(self.filename + '.npy'):
            os.remove(self.filename + '.npy')


if __name__ == '__main__':
    # Example Usage
    storage = MTSStorage("my_mts_data")

    # Create some dummy data
    mts_data_1 = np.random.rand(10, 100, 3)  # 10 series, 100 time points, 3 dimensions
    mts_data_2 = np.random.rand(5, 150, 2)   # 5 series, 150 time points, 2 dimensions
    mts_data_3 = np.random.rand(1000, 200, 4)   # 8 series, 200 time points, 4 dimensions


    coords_1 = {'pca': np.random.rand(10, 2), 'tsne': np.random.rand(10, 2)}
    labels_1 = {'cluster_ids': np.random.randint(0, 3, 10)}
    label_names_1 = {'cluster_ids': {0: 'Group A', 1: 'Group B', 2: 'Group C'}}
    dimensions_1 = ['sensor_x', 'sensor_y', 'sensor_z']


    coords_3 = np.random.rand(1000, 2)

    storage.add_mts_multi(
        name="mts_1",
        mts=mts_data_1,
        dimensions=dimensions_1,
        coords=coords_1,
        labels=labels_1,
        # label_names=label_names_1
    )

    storage.add_mts_multi(
        name="mts_2",
        mts=mts_data_2,
        coords={'umap': np.random.rand(5, 2)},
        labels={'category': np.random.randint(0, 2, 5)}
    )

    storage.add_mts(
        name="mts_3",
        mts=mts_data_3,
        projection=coords_3
    )

    storage.save()
    print("Data saved.")

    storage_loaded = MTSStorage("my_mts_data")
    storage_loaded.load()
    print("Data loaded.")

    # Access loaded data
    loaded_mts_1 = storage_loaded.objects['mts_1']
    print(f"Loaded MTS object name: {loaded_mts_1.name}")
    print(f"Loaded MTS data shape: {loaded_mts_1.mts.shape}")
    if loaded_mts_1.coords:
        print(f"Loaded coords keys: {loaded_mts_1.coords.keys()}")
    if loaded_mts_1.labels:
        print(f"Loaded label keys: {loaded_mts_1.labels.keys()}")

    # storage_loaded.delete()
    # print("Storage file deleted.")