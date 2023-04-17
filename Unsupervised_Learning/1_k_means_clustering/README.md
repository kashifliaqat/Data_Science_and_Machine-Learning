# K-Means Clustering

## Task
Cluster mall customers based on their annual income and spending score using k-means clustering algorithm, and then interpret the resulting clusters to gain insights into the purchasing behavior. 

## Dataset
The [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) is a dataset that contains information about customers of a mall, and it is often used for customer segmentation or clustering analysis. This dataset can be useful for businesses, marketers, or analysts who are interested in understanding their customers' behavior and preferences to optimize their marketing strategies or product offerings.

Explanation of each variable in the Dataset:

1. CustomerID: unique identifier for each customer.

2. Gender: the gender of the customer, either Male or Female.

3. Age: the age of the customer.

4. Annual Income (k$): the annual income of the customer in thousands of dollars.

5. Spending Score (1-100): a score assigned by the mall based on customer behavior and spending nature. 

## The K-Means Clustering Algorithm
K-means clustering is an unsupervised machine learning technique that aims to partition a dataset into K clusters, where K is a user-defined hyperparameter. The algorithm works by iteratively assigning each data point to the closest cluster centroid and then recomputing the centroid based on the new assignment. The process continues until the assignments converge, i.e., they no longer change.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/kmeans.png" alt="K-Means Clustering" width="500 height="300">


Image Source: [javaTpoint](https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-learning)



The working of the K-means algorithm can be explained as follows:

1. Initialize K cluster centroids randomly
2. Assign each data point to the nearest centroid
3. Recalculate the centroids based on the mean of the points assigned to each cluster
4. Repeat steps 2 and 3 until convergence (i.e., no data points are reassigned to a different cluster)

### Mathematical Formulation 

Mathematically, the K-means algorithm can be formulated as follows:

1. Initialize K cluster centroids:
$$c_1, c_2, ..., c_K$$
2. Assign each data point to the nearest centroid:
$$argmin_j ||x_i - c_j||^2$$
where $x_i$ is the i-th data point and $c_j$ is the j-th centroid.
3. Recalculate the centroids based on the mean of the points assigned to each cluster:
$$c_j = (1/|S_j|) * sum_{i in S_j} x_i$$
where $S_j$ is the set of data points assigned to cluster j.

4. Repeat steps 2 and 3 until convergence.

The objective function of K-means clustering is given by the sum of squared distances between each data point and its assigned centroid:
$$J(c_1, ..., c_K) = sum_{i=1}^n ||x_i - c_{k(i)}||^2$$

where $k(i)$ is the index of the centroid that the i-th data point is assigned to.

The goal of the algorithm is to find the centroids that minimize this objective function.

## References/Resources 
- [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [K-Means Clustering Algorithm - javaTpoint](https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-learning)
