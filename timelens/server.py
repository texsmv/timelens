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
        mts_object = storage.get_mts(objectName,)
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
