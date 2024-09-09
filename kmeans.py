"""
K-Means Clustering Algorithm:

The K-Means algorithm is an unsupervised machine learning technique used for clustering data points into 'k'
clusters based on their similarities. The algorithm works as follows:

1. Initialisation:
    - Randomly initialise 'k' centroids. The centroids are representative points for each data.

2. Assignment:
    - Assign each data point to the nearest centroid based on a distance metric, typically the Euclidean
      distance.

3. Update:
    - Recalculate the centroids based on the mean of the data points assigned to each cluster.

4. Repeat:
    - Repeat the assignment and update steps for a specific number of iterations or until convergence.

Approach to the code:
- The code begins by defining necessary functions for calculating Euclidean distance, reading data from a CSV file,
  finding the closest centroid to each data point, initialising centroids, performing the k-means algorithm, plotting
  clusters, and printing results.
- The 'read data' file reads data from the provided CSV file, skipping the header row and converting the data to a
  numpy array.
- The 'k-means' function implements the k-means algorithm by initialising the centroids, assigning data points to
  clusters, updating centroids, and repeating the process for a specified number of iterations.
- Finally, the user is prompted to enter the number of clusters and iterations and the k-means algorithm is executed
  with the provided parameters.

References:

1. Gajawada, S.K, 2019, "K-Means Clustering with Math: Common Unsupervised learning technique for data analysis", Medium,
   viewed 15 February 2024, retrieved from https://towardsdatascience.com/k-means-clustering-for-beginners-2dc7b2994a4.

2. Sharma, P, 2024, "The Ultimate Guide to K-Means Clustering: Definition, Methods and Applications",  Analytics Vidhya,
   viewed 17 March 2024, retrieved from https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/.

3. Ulloa, P, 2017, "Understanding Cluster Analysis K-Means and Euclidean Distance", linkedin, viewed 01 March 2024, retrieved from
   https://www.linkedin.com/pulse/understanding-cluster-analysis-k-means-euclidean-paul-ulloa-mba?trk=portfolio_article-card_title.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

# Define a function that computes the distance between two data points
def euclidean_distance(point1, point2):
    """
    Compute the Euclidean distance between two points.

    :param point1: A tuple containing the coordinates of the first point.
    :param point2: A tuple containing the coordinates of the second point.
    :return: A float value of the Euclidean distance between the two points.
    """
    # Separate the x and y coordinates of point1
    x1, y1 = point1[0], point1[1]

    # Separate the x and y coordinates of point2
    x2, y2 = point2[0], point2[1]

    # Calculate the Euclidean distance using nested loops
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance

# Define a function that reads data from a CSV file
def read_data(file_path):
    """
    Read data from a CSV file.

    :param file_path: The path of the CSV file.
    :return: An array containing the data read from the file.
    """
    data = []

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header row
        for row in reader:
            data.append([float(row[1]), float(row[2])])

    return np.array(data)

# Define a function that finds the closest centroid to each point out of all the centroids
def find_closest_centroid(data_point, centroids):
    """
    Find the index of the closest centroid to a given data point.

    :param data_point: The coordinates of the data point.
    :param centroids: An array containing centroids.
    :return: The index of the closest centroid.
    """
    distances = [euclidean_distance(data_point, centroid) for centroid in centroids]
    return np.argmin(distances)

# Write a function to visualize the clusters
def plot_all_iterations(data, all_centroids, all_clusters):
    """
    Plot the clusters and centroids.

    :param data: An array containing the data points.
    :param centroids: An array containing the centroids.
    :param clusters: An array containing the cluster assignments.
    """
    num_iterations = len(all_centroids)
    fig, axs = plt.subplots(num_iterations, 1, figsize=(12, 6 * num_iterations))

    for i in range(num_iterations):
        ax = axs[i]
        if i < len(all_clusters):  # Check if index is within range
            clusters = all_clusters[i]
            if len(clusters) > 0:
                ax.scatter(data[:, 0], data[:, 1], c=clusters, cmap='rainbow', marker='o')
                centroids = all_centroids[i]
                ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200)
                ax.set_title(f'Iteration {i + 1}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')

    plt.tight_layout()
    plt.show()


# Write the initialization procedure
def initialize(data, k):
    """
    Initialize the centroids randomly.

    :param data: An array containing the data points.
    :param k: The number of clusters.
    :return: The array containing the initialized centroids.
    """
    centroids = data[np.random.choice(len(data), k, replace=False)]
    return centroids

# Implement the k-means algorithm
def k_means(data, k, iterations):
    """
    Perform the K-Means clustering algorithm.

    :param data: An array containing the data points.
    :param k: The number of clusters.
    :param iterations: The number of iterations.
    :return:
        numpy.array: An array containing the final centroids.
        numpy.array: An array containing the cluster assignments.
    """
    centroids = initialize(data, k)
    all_clusters = []
    all_centroids = [centroids.copy()]

    for _ in range(iterations):
        clusters = np.array([find_closest_centroid(point, centroids) for point in data])
        all_clusters.append(clusters.copy())
        for i in range(k):
            cluster_points = data[clusters == i]
            centroids[i] = np.mean(cluster_points, axis=0)
        all_centroids.append(centroids.copy())
    return all_centroids, all_clusters

# Print out the results
def print_results(clusters):
    """
    Print the results of the clustering.

    :param clusters: An array containing the cluster assignments.
    :param data: The original data points.
    """
    for i, clusters in enumerate(all_clusters):
        print(f"Iteration {i + 1}:")
        for j in range(len(np.unique(clusters))):
            cluster_points = data[clusters == j]
            if len(cluster_points) > 0:
                print(f"\nCluster {j + 1}:")
                print(f"Number of countries: {len(cluster_points)}")
                # Print list of countries (if available)
                print(f"Countries: {', '.join(cluster_points.index)}" if hasattr(cluster_points, 'index') else "")
                print(f"Mean Life Expectancy: {np.nanmean(cluster_points[:, 0]):.2f}")
                print(f"Mean Birth Rate: {np.nanmean(cluster_points[:, 1]):.2f}")
        print("\n")

# Define file path
file_path = 'dataBoth.csv'

# Read data from CSV file
data = read_data(file_path)

# Prompt the user to enter parameters
num_clusters = int(input("\nEnter the number of clusters: "))
num_iterations = int(input("Enter the number of iterations: "))

# Run the k-means algorithm
all_centroids, all_clusters = k_means(data, num_clusters, num_iterations)

# Print results
print_results(all_clusters)

# Plot clusters
plot_all_iterations(data, all_centroids, all_clusters)