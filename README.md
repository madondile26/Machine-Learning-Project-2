# K-Means Clustering Implementation

## Overview

This project implements the K-Means clustering algorithm, an unsupervised machine learning technique used to group data points into `k` clusters based on their similarities. The algorithm iteratively assigns each data point to the nearest cluster centroid and updates the centroids until convergence or a specified number of iterations is reached.

## Features

The script performs the following tasks:
- **Calculate Euclidean Distance**: Computes the distance between data points to determine cluster assignments.
- **Data Reading**: Reads data from a CSV file containing two features (e.g., life expectancy and birth rate).
- **Centroid Initialization**: Randomly selects initial centroids for the clustering process.
- **K-Means Algorithm**: Iteratively assigns data points to clusters, updates centroids, and repeats the process for a specified number of iterations.
- **Result Printing**: Outputs the number of data points in each cluster, along with their mean values.
- **Visualization**: Plots the clusters and centroids at each iteration to visualize the clustering process.

## Requirements

- Python 3.x
- `numpy` for numerical operations
- `matplotlib` for plotting
- `csv` for reading data from CSV files

You can install the required packages using pip:
```bash
pip install numpy matplotlib
