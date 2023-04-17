# # Visualizing: K-Means VS DBSCAN

## Task
Visualize the K-Means and DBSCAN clustering algorithms uisng `make_blobs`, `make_moons`, and `make_circles` datasets. 

## Dataset
Data is generated using the `make_blobs`, `make_moons`, and `make_circles` function from the `sklearn.datasets` module.

The datasets blobs, moon, and circles are synthetic datasets commonly used in machine learning research to test and compare algorithms' performance.

1. **Blobs:** This dataset consists of randomly generated Gaussian blobs, i.e., groups of points with a Gaussian distribution around them. The blobs are usually well-separated and have different numbers of clusters, making it a good dataset to test clustering algorithms.

2. **Moon:** This dataset consists of two half-moon shapes, with one moon above the other. It is commonly used to test algorithms' ability to separate non-linearly separable data.

3. **Circles:** This dataset consists of two concentric circles, with one circle inside the other. It is also commonly used to test algorithms' ability to separate non-linearly separable data.

## DBSCAN vs K-Means
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) and K-means are both clustering algorithms that are used in unsupervised machine learning.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/db_vs_K.PNG" alt="DBSCAN vs K-Means" width="500 height="300">


Image Source: [NSHipster - GitHub](https://github.com/NSHipster/DBSCAN)

The main difference between DBSCAN and K-means is that DBSCAN is a density-based clustering algorithm, while K-means is a centroid-based clustering algorithm.

K-means tries to minimize the sum of squared distances between data points and their assigned centroids. It starts by randomly initializing k cluster centers and then iteratively assigns each data point to the nearest cluster center until the cluster centers no longer change. The result is that each data point belongs to the cluster whose centroid it is closest to.

DBSCAN, on the other hand, groups together data points that are close together in high-density regions and identifies noise points that lie in low-density regions. DBSCAN does not require the number of clusters to be specified in advance, and it can find clusters of arbitrary shape.

Another difference between the two algorithms is that DBSCAN can handle outliers, while K-means cannot. DBSCAN defines data points that are not part of any cluster as noise points, whereas K-means assigns every data point to a cluster.

In summary, K-means is a centroid-based algorithm that requires the number of clusters to be specified in advance and can only handle data that can be represented by a mean vector. DBSCAN is a density-based algorithm that can identify clusters of arbitrary shape and can handle outliers.

## References/Resources 
- [NSHipster/DNSCAN - GitHub](https://github.com/NSHipster/DBSCAN)
