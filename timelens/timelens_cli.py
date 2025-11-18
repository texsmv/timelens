# timelens/timelens_cli.py

import argparse
# Import the new server initialization function
from timelens.server import init_server 
import os

def main():
    parser = argparse.ArgumentParser(description="Timelens: Visualize Time Series Data")
    parser.add_argument("pod_path", help="Path to the TSPod file (.npz)")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port number (default: 5000)")

    args = parser.parse_args()

    # Check if the .npz file exists
    if not os.path.exists(args.pod_path):
        print(f"Error: TSPod file not found at {args.pod_path}")
        return

    print(f"Starting Timelens server with pod: {args.pod_path}")
    
    # Call the new init_server function
    init_server(args.pod_path, host=args.host, port=args.port)

if __name__ == "__main__":
    main()