# timelens/server.py
from flask import Flask, jsonify, request
from flask_cors import CORS
# We now import our new TSPod class
from timelens.storage import TSPod 
from timelens.metrics import run_numerical_tests, run_categorical_tests
from timelens.clustering import get_clustering_result, estimate_dbscan_eps 

# Global variable to hold our single loaded TSPod instance
tspod = None

app = Flask(__name__)
# Enable Cross-Origin Resource Sharing
CORS(app)

@app.route("/info", methods=['POST'])
def get_pod_info():
    """
    Endpoint to get high-level metadata about the loaded TSPod.
    """
    global tspod
    if tspod is None:
        return jsonify({"error": "TSPod is not loaded on the server."}), 500
    
    # Use the get_info() method we created for the TSPod class
    info_dict = tspod.get_info()
    return jsonify(info_dict)


@app.route("/data", methods=['POST'])
def get_pod_data():
    """
    Endpoint to get the main data payload from the TSPod.
    Accepts an optional 'max_series' parameter in the JSON body to sample the data.
    """
    global tspod
    if tspod is None:
        return jsonify({"error": "TSPod is not loaded on the server."}), 500

    # Get 'max_series' from JSON body
    data = request.get_json() or {}
    max_series = data.get('max_series', None)
    print(f"Received max_series: {max_series}")
    
    if max_series is not None:
        try:
            max_series = int(max_series)
            print(f"Converted max_series to int: {max_series}")
        except (ValueError, TypeError):
            return jsonify({
                "error": f"Invalid 'max_series' parameter: '{max_series}'. Must be an integer."
            }), 400
    
    # Use the get_data_payload() method with the optional sampling parameter
    data_payload = tspod.get_data_payload(max_series=max_series)
    return jsonify(data_payload)


@app.route("/statistical_tests/numerical", methods=['POST'])
def numerical_statistical_tests():
    """
    Endpoint to perform statistical tests on numerical variables.
    Expects JSON payload with 'values' (list of numerical values) and 'group_labels' (list of 0s and 1s).
    """
    global tspod
    if tspod is None:
        return jsonify({"error": "TSPod is not loaded on the server."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        values = data.get('values')
        group_labels = data.get('group_labels')
        variable_name = data.get('variable_name', 'Unknown Variable')
        
        if values is None or group_labels is None:
            return jsonify({
                "error": "Missing required fields: 'values' and 'group_labels'"
            }), 400
        
        # Validate that values and group_labels have the same length
        if len(group_labels) != len(values):
            return jsonify({
                "error": f"group_labels length ({len(group_labels)}) does not match values length ({len(values)})"
            }), 400
        
        # Run statistical tests
        results = run_numerical_tests(values, group_labels)
        results['variable_name'] = variable_name
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": f"Error performing numerical statistical tests: {str(e)}"}), 500


@app.route("/statistical_tests/categorical", methods=['POST'])
def categorical_statistical_tests():
    """
    Endpoint to perform statistical tests on categorical variables.
    Expects JSON payload with 'categories' (list of category values) and 'group_labels' (list of 0s and 1s).
    """
    global tspod
    if tspod is None:
        return jsonify({"error": "TSPod is not loaded on the server."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        categories = data.get('categories')
        group_labels = data.get('group_labels')
        variable_name = data.get('variable_name', 'Unknown Variable')
        
        if categories is None or group_labels is None:
            return jsonify({
                "error": "Missing required fields: 'categories' and 'group_labels'"
            }), 400
        
        # Validate that categories and group_labels have the same length
        if len(group_labels) != len(categories):
            return jsonify({
                "error": f"group_labels length ({len(group_labels)}) does not match categories length ({len(categories)})"
            }), 400
        
        # Run statistical tests
        results = run_categorical_tests(categories, group_labels)
        results['variable_name'] = variable_name
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": f"Error performing categorical statistical tests: {str(e)}"}), 500


@app.route("/clustering", methods=['POST'])
def perform_clustering():
    """
    Endpoint to perform clustering on 2D projection data.
    Expects JSON payload with:
    - 'data': List of [x, y] coordinates for each point
    - 'algorithm': Either 'kmeans' or 'dbscan'
    - Algorithm-specific parameters (optional)
    
    For K-means:
    - 'n_clusters': Number of clusters (default: 3)
    - 'random_state': Random seed (default: 42)
    
    For DBSCAN:
    - 'eps': Maximum distance between samples (default: auto-estimated)
    - 'min_samples': Minimum samples in neighborhood (default: 5)
    """
    global tspod
    if tspod is None:
        return jsonify({"error": "TSPod is not loaded on the server."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        projection_data = data.get('data')
        algorithm = data.get('algorithm')
        
        if projection_data is None or algorithm is None:
            return jsonify({
                "error": "Missing required fields: 'data' and 'algorithm'"
            }), 400
        
        # Validate projection data format
        if not isinstance(projection_data, list) or len(projection_data) == 0:
            return jsonify({"error": "Data must be a non-empty list"}), 400
        
        for point in projection_data:
            if not isinstance(point, list) or len(point) != 2:
                return jsonify({
                    "error": "Each data point must be a list of exactly 2 coordinates [x, y]"
                }), 400
        
        # Prepare algorithm parameters
        kwargs = {}
        
        if algorithm.lower() == 'kmeans':
            kwargs['n_clusters'] = data.get('n_clusters', 3)
            kwargs['random_state'] = data.get('random_state', 42)
        elif algorithm.lower() == 'dbscan':
            # If eps is not provided, estimate it
            if 'eps' not in data:
                try:
                    estimated_eps = estimate_dbscan_eps(projection_data)
                    kwargs['eps'] = estimated_eps
                except Exception as e:
                    # Fall back to default if estimation fails
                    kwargs['eps'] = 0.5
            else:
                kwargs['eps'] = data.get('eps')
            kwargs['min_samples'] = data.get('min_samples', 5)
        
        # Perform clustering
        clustering_result = get_clustering_result(projection_data, algorithm, **kwargs)
        
        # Add metadata about the clustering
        result = {
            'clustering_result': clustering_result,
            'algorithm': algorithm,
            'parameters': kwargs,
            'n_points': len(projection_data)
        }
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({"error": f"Invalid parameters: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error performing clustering: {str(e)}"}), 500


def init_server(pod_path: str, host: str = "127.0.0.1", port: int = 5000):
    """
    Loads the TSPod from a file and starts the Flask server.
    """
    global tspod
    print(f"🚀 Loading TSPod from '{pod_path}'...")
    try:
        # Load the single TSPod object into our global variable
        tspod = TSPod.load(pod_path)
        print("✅ TSPod loaded successfully.")
        print(f"   - Name: {tspod.name}")
        print(f"   - Shape: {tspod.shape}")
        print(f"🌍 Starting server at http://{host}:{port}")
        
        # Start the Flask web server
        app.run(host=host, port=port, debug=False)
        
    except FileNotFoundError:
        print(f"❌ Error: TSPod file not found at '{pod_path}'")
    except Exception as e:
        print(f"❌ An unexpected error occurred during server initialization: {e}")


if __name__ == "__main__":
    # This block allows you to test the server by running `python -m timelens.server`
    # Make sure you have a .npz file created by TSPod.save() available.
    example_pod_path = 'my_weather_data.npz' # Change this to your pod file
    init_server(example_pod_path, host="127.0.0.1", port=5000)