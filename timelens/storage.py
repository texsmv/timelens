import numpy as np
from typing import Dict, List, Optional, Any
from timelens.utils import project_mts


class TSPod:
    """
    A self-contained archive for a single multivariate time series (MTS) dataset.

    This class stores the core time series data, its metadata (like dimension names),
    a 2D projection, and any associated variables describing each individual series.

    Attributes:
        name (str): The name for the dataset.
        data (np.ndarray): The core MTS data with shape (N, T, D).
        dimension_names (List[str]): A list of names for the D dimensions.
        projection (np.ndarray): A 2D projection array of shape (N, 2).
        series_variables (Dict[str, Dict[str, Any]]): A dictionary to hold variables
            describing each of the N series.
    """
    def __init__(self,
                 name: str,
                 data: np.ndarray,
                 dimension_names: List[str],
                 projection: Optional[np.ndarray] = None):
        """
        Initializes the TSPod.
        """
        # --- Validate Inputs ---
        if not name or not isinstance(name, str):
            raise ValueError("`name` must be a non-empty string.")
        if not isinstance(data, np.ndarray) or data.ndim != 3:
            raise ValueError("`data` must be a 3D NumPy array of shape (N, T, D).")

        n_series, _, n_dims = data.shape

        if not isinstance(dimension_names, list) or len(dimension_names) != n_dims:
            raise ValueError(f"`dimension_names` must be a list of {n_dims} strings.")

        # --- Assign Attributes ---
        self.name = name
        self.data = data
        self.dimension_names = dimension_names
        self.series_variables: Dict[str, Dict[str, Any]] = {}

        # --- Handle Projection ---
        if projection is None:
            print("ℹ️ No projection provided. Computing a 2D projection using PCA...")
            self.projection = project_mts(self.data)
        else:
            if projection.shape[0] != n_series:
                 raise ValueError(f"Provided projection must have {n_series} rows, but got {projection.shape[0]}.")
            self.projection = projection

    @property
    def shape(self):
        """Returns the (N, T, D) shape of the time series data."""
        return self.data.shape

    def __repr__(self) -> str:
        """Provides a concise string representation of the object."""
        n_series, n_timesteps, n_dims = self.shape
        var_count = len(self.series_variables)
        return (f"<TSPod name='{self.name}' "
                f"shape=({n_series}, {n_timesteps}, {n_dims}) "
                f"variables={var_count}>")

    def add_numerical_variable(self, var_name: str, values: np.ndarray):
        """
        Adds a numerical variable describing the N series.
        """
        if values.shape != (self.shape[0],):
            raise ValueError(f"Numerical values array must have shape ({self.shape[0]},).")

        self.series_variables[var_name] = {
            'type': 'numerical',
            'values': values
        }
        print(f"✅ Added numerical variable: '{var_name}'")

    def add_categorical_variable(self, var_name: str, values: np.ndarray, labels: Dict[int, str]):
        """
        Adds a categorical variable describing the N series.
        """
        if values.shape != (self.shape[0],):
            raise ValueError(f"Categorical values array must have shape ({self.shape[0]},).")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("Categorical values must be an array of integers.")

        self.series_variables[var_name] = {
            'type': 'categorical',
            'values': values,
            'labels': labels
        }
        print(f"✅ Added categorical variable: '{var_name}'")

    def save(self, file_path: str):
        """
        Saves the entire pod to a compressed .npz file.
        """
        if not file_path.endswith('.npz'):
            file_path += '.npz'

        payload = {
            'name': np.array(self.name),
            'data': self.data,
            'dimension_names': np.array(self.dimension_names, dtype=object),
            'projection': self.projection,
            'series_variables': np.array(self.series_variables)
        }

        np.savez_compressed(file_path, **payload)
        print(f"💾 Pod saved successfully to '{file_path}'")

    @classmethod
    def load(cls, file_path: str) -> 'TSPod':
        """
        Loads a pod from a .npz file.
        """
        with np.load(file_path, allow_pickle=True) as loaded_data:
            # .item() extracts the value from a 0-d object array
            name = loaded_data['name'].item()
            data = loaded_data['data']
            dimension_names = list(loaded_data['dimension_names'])
            projection = loaded_data['projection']

            # Instantiate the class
            instance = cls(name, data, dimension_names, projection=projection)

            # Load series variables if they exist
            if 'series_variables' in loaded_data:
                instance.series_variables = loaded_data['series_variables'].item()

        print(f"📂 Pod loaded successfully from '{file_path}'")
        return instance
    
    def get_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing a summary of the pod's metadata.
        """
        n_series, n_timesteps, n_dims = self.shape
        
        categorical_vars = [
            name for name, var_meta in self.series_variables.items() 
            if var_meta['type'] == 'categorical'
        ]
        numerical_vars = [
            name for name, var_meta in self.series_variables.items() 
            if var_meta['type'] == 'numerical'
        ]
        
        return {
            "name": self.name,
            "n_series": n_series,
            "n_timesteps": n_timesteps,
            "n_dims": n_dims,
            "categorical_variables": categorical_vars,
            "numerical_variables": numerical_vars
        }

    def get_data_payload(self, max_series: Optional[int] = None) -> Dict[str, Any]:
        """
        Prepares the pod's data for server transmission with a clear and organized
        structure, allowing for optional sampling.

        Args:
            max_series (Optional[int]): The maximum number of series to return.
                                        If None, all series are returned.

        Returns:
            A dictionary formatted for use as a JSON API response.
        """
        n_series, _, _ = self.shape
        
        # Determine which series indices to use (all or a random sample)
        indices = np.arange(n_series)
        if max_series is not None and max_series < n_series:
            print(f"ℹ️ Sampling {max_series} out of {n_series} series.")
            indices = np.random.choice(indices, max_series, replace=False)

        # Apply sampling indices to the core data
        sampled_data = self.data[indices]
        sampled_projection = self.projection[indices]
        
        # --- NEW: Process variables into separate, self-contained dictionaries ---
        numerical_payload = {}
        categorical_payload = {}
        
        for name, var_meta in self.series_variables.items():
            sampled_values = var_meta['values'][indices].tolist()
            
            if var_meta['type'] == 'numerical':
                numerical_payload[name] = sampled_values
            
            elif var_meta['type'] == 'categorical':
                # Convert integer keys to strings for JSON compatibility
                string_key_labels = {str(k): v for k, v in var_meta['labels'].items()}
                categorical_payload[name] = {
                    "values": sampled_values,
                    "labels": string_key_labels
                }

        # Construct the final, more organized payload
        payload = {
            "data": sampled_data.flatten().tolist(),
            "shape": sampled_data.shape,
            "dimensions": self.dimension_names,
            "projection": sampled_projection.flatten().tolist(),
            "numerical_variables": numerical_payload,
            "categorical_variables": categorical_payload
        }
        return payload
