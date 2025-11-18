This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: **/*.py
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

## Additional Info

# Directory Structure
```
timelens/
  datasets.py
  server.py
  storage.py
  timelens_cli.py
  utils.py
setup.py
```

# Files

## File: timelens/datasets.py
```python
import pandas as pd
import numpy as np
import os

from sktime.datasets import load_from_tsfile

def loadFuncionalModel(n):
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    df = pd.read_csv(os.path.join(dirname,'datasets/outliers/model{}.csv'.format(n)))
    data = df.to_numpy()
    return data
    
def loadWafer():
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    X_train, y_train = load_from_tsfile(os.path.join(dirname, 'datasets/Wafer/Wafer_TRAIN.ts'), return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(os.path.join(dirname,'datasets/Wafer/Wafer_TEST.ts'), return_data_type="numpy3d")
    
    y_train = np.array([int(y_train[i]) for i in range(len(y_train))])
    y_test = np.array([int(y_test[i]) for i in range(len(y_test))])
    
    return X_train, y_train, X_test, y_test

def loadWeather():
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    X_train, y_train = load_from_tsfile(os.path.join(dirname, 'datasets/Wafer/Wafer_TRAIN.ts'), return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(os.path.join(dirname,'datasets/Wafer/Wafer_TEST.ts'), return_data_type="numpy3d")
    
    return X_train, y_train, X_test, y_test


# Returns data with the shape NxDxT
def loadNatops():
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    X_train, y_train = load_from_tsfile(os.path.join(dirname, 'datasets/NATOPS/NATOPS_TRAIN.ts'), return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(os.path.join(dirname,'datasets/NATOPS/NATOPS_TEST.ts'), return_data_type="numpy3d")
    
    return X_train, y_train, X_test, y_test

def loadSelfRegulationSCP2():
    dirname = os.path.dirname(__file__)
    dirname = os.path.abspath(os.path.join(dirname, os.pardir))
    X_train, y_train = load_from_tsfile(os.path.join(dirname, 'datasets/SelfRegulationSCP2/SelfRegulationSCP2_TRAIN.ts'), return_data_type="numpy3d")
    X_test, y_test = load_from_tsfile(os.path.join(dirname,'datasets/SelfRegulationSCP2/SelfRegulationSCP2_TEST.ts'), return_data_type="numpy3d")
    
    return X_train, y_train, X_test, y_test
```

## File: timelens/timelens_cli.py
```python
# timelens_cli.py
import argparse
from timelens.server import initServer
import os

def main():
    parser = argparse.ArgumentParser(description="Timelens: Visualize Time Series Data")
    parser.add_argument("storage_path", help="Path to the MTSStorage file (.npy)")
    parser.add_argument("--host", default="127.0.0.1", help="Host address for the server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port number for the server (default: 5000)")

    args = parser.parse_args()

    if not os.path.exists(args.storage_path + '.npy'): # Check if .npy exists
        print(f"Error: Storage file not found at {args.storage_path + '.npy'}")
        return

    print(f"Starting Timelens server with storage: {args.storage_path}")
    print(f"Host: {args.host}, Port: {args.port}")

    initServer(args.storage_path, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
```

## File: timelens/utils.py
```python
from sklearn.decomposition import PCA


# Projects multivariate time series of shape N, T, D to N, 2
# 
# First reshape the input to N, T*D, then project to N, 2
def project_mts(X, reducer=None):
    N, T, D = X.shape
    X = X.reshape(N, T*D)
    if reducer is None:
        reducer = PCA(n_components=2)
    reducer.fit(X)
    return reducer.transform(X)
```

## File: setup.py
```python
from setuptools import setup, find_packages

setup(
    name='timelens',  # Your package name
    version='0.1.0',  # Your package version
    packages=find_packages(), # Automatically find your packages
    entry_points={
        'console_scripts': [
            'timelens=timelens.timelens_cli:main', # This line is the important part
        ],
    },
    install_requires=[ # List your dependencies here
        'flask',
        'flask-cors',
        'numpy',
        'argparse' # Make sure argparse is included if you are using it in timelens_cli.py
    ],
)
```

## File: timelens/server.py
```python
# mts_vis/server.py
import re
from flask import Flask, jsonify, request
from flask_cors import CORS
from timelens.storage import MTSStorage  # Import MTSStorage from within the package
import numpy as np
from flask import jsonify

# path ='mts_datasets' # Remove unused path variable

storage = None # Remove global storage initialization
objects = None # Remove global objects initialization


app = Flask(__name__)
CORS(app)

@app.route("/objectsInfo", methods=['POST'])
def objectsInfo():
    global objects
    if storage is None:
        return jsonify({'names': []}) # Return empty list if no storage loaded
    return jsonify({'names': list(storage.objects.keys())}) # Access keys from storage.objects


@app.route("/object", methods=['POST'])
def object():
    objectName = request.get_json()["name"]
    max_windows = request.get_json()["max_windows"]


    global storage
    if storage is None or objectName not in storage.objects:
        return jsonify({'status': 'Error', 'message': 'Object not found or storage not loaded.'}) # More informative error
    else:
        mts_object = storage.get_mts(objectName, None)
        N, T, D = mts_object.mts.shape

        if max_windows < N:
            mts_object = storage.get_mts(objectName, max_windows)
            N, T, D = mts_object.mts.shape

        resp_map = {}
        resp_map['data'] = mts_object.mts.flatten().tolist()
        resp_map['shape'] = mts_object.mts.shape

        if mts_object.coords is not None: 
            resp_map['coords'] = {}
            for k, v in mts_object.coords.items():
                resp_map['coords'][k] = v.flatten().tolist()

        if mts_object.labels is not None : # Access labels from mts_object
            resp_map['labels'] = {}
            for k, v in mts_object.labels.items():
                resp_map['labels'][k] = v.flatten().tolist()

        if mts_object.label_names is not None: # Access label_names from mts_object
            resp_map['labelsNames'] = mts_object.label_names
        
        elif mts_object.labels is not None: # Fallback to create label names from unique values if not provided
            resp_map['labelsNames'] = {}
            for k, v in mts_object.labels.items():
                labls = np.unique(v.flatten())
                resp_map['labelsNames'][k] = { str(int(l)):int(l) for l in labls } # Ensure keys are strings

        else:
            resp_map['labelsNames'] = {} # Ensure labelsNames is always in response, even if empty


        if mts_object.dimensions is not None: # Check if dimensions exist and is not empty
            resp_map['dimensions'] = mts_object.dimensions
        else:
            resp_map['dimensions'] = [f"dimension_{i}" for i in range(D)] # Default dimension names

        return jsonify(resp_map)



def initServer(storage_path, host = "127.0.0.1", port=5000): # Modified to accept storage_path
    global storage
    storage = MTSStorage(storage_path) # Use storage_path to create MTSStorage instance
    storage.load()

    global objects
    objects = storage.objects # Access objects from the storage instance

    CORS(app)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    # Example of how to launch the server programmatically (without CLI)
    # For testing purposes you can still run server.py directly
    # but normally you would use timelens_cli.py
    example_storage_path = 'mts_datasets/example_storage' # Example path, adjust as needed
    initServer(example_storage_path, host="127.0.0.1", port=5000)
```

## File: timelens/storage.py
```python
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
```
