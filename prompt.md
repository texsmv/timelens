# timelens: Time Series Visualization and Exploration Package

**timelens** is a Python package designed to help you visualize and explore time series data, especially through the lens of dimensionality reduction techniques.

## Key Functionality

- **Time Series Data Storage**: Provides a way to store your time series datasets along with associated information like labels, dimensionality reduction projections (coordinates), and dimension/label names. This stored data can be saved to and loaded from files.

- **Interactive Visualization Server**: Includes a built-in server that serves this stored time series data. This server is designed to communicate with client applications (like a Flutter app) to enable interactive visualization of your time series and their dimensionality-reduced representations.

- **Data Serving for Visualization**: The server exposes an API that allows client applications to request and receive time series data, object names, and related metadata. This enables the client to display and interact with the data visually, facilitating exploration and analysis.

- **Dataset Loading (Optional)**: Optionally includes tools for loading common time series datasets, making it easier to get started with visualization and exploration.

In essence, **timelens** helps bridge the gap between your Python-based time series analysis and interactive visual exploration in a separate application, by providing structured data storage and a server to deliver that data for visualization.
