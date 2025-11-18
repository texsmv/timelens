# create_test_pod.py

import numpy as np
# Make sure your TSPod class is accessible. 
# This import assumes it's in timelens/storage.py
from timelens.storage import TSPod 

def create_test_data():
    """Generates and saves a sample TSPod for testing."""
    print("📦 Creating sample TSPod data...")
    
    # 1. Define data parameters
    n_series = 100
    n_timesteps = 200
    n_dims = 5
    
    # 2. Create the main data array
    mts_data = np.random.randn(n_series, n_timesteps, n_dims)
    dim_names = [f'Sensor_{i+1}' for i in range(n_dims)]
    
    # 3. Instantiate the TSPod (projection will be auto-generated)
    pod = TSPod(
        name="SampleSensorData",
        data=mts_data,
        dimension_names=dim_names
    )
    
    # 4. Add some series variables for testing
    pod.add_numerical_variable(
        var_name="temperature",
        values=np.random.uniform(15, 30, size=n_series)
    )
    pod.add_categorical_variable(
        var_name="status",
        values=np.random.randint(0, 3, size=n_series),
        labels={0: 'offline', 1: 'online', 2: 'error'}
    )
    
    # 5. Save the pod to a file
    file_path = "test_pod.npz"
    pod.save(file_path)
    print(f"\n✅ Test data saved successfully to '{file_path}'")

if __name__ == "__main__":
    create_test_data()