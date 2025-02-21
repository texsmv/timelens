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