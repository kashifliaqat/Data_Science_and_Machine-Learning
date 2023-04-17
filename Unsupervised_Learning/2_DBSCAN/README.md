# Density-Based Spatial Clustering of Applications with Noise (DBSCAN) 

## Task
Cluster mall customers based on their annual income and spending score using DBSCAN algorithm, and then interpret the resulting clusters to gain insights into the purchasing behavior. 

## Dataset
The [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) is a dataset that contains information about customers of a mall, and it is often used for customer segmentation or clustering analysis. This dataset can be useful for businesses, marketers, or analysts who are interested in understanding their customers' behavior and preferences to optimize their marketing strategies or product offerings.

Explanation of each variable in the Dataset:

1. CustomerID: unique identifier for each customer.

2. Gender: the gender of the customer, either Male or Female.

3. Age: the age of the customer.

4. Annual Income (k$): the annual income of the customer in thousands of dollars.

5. Spending Score (1-100): a score assigned by the mall based on customer behavior and spending nature. 

## The DBSCAN Algorithm
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups together points that are closely packed together in high-density regions, while points in low-density regions are considered outliers or noise.

The algorithm works by defining a distance threshold (eps) and a minimum number of points (minPts) to form a dense region. Starting from an arbitrary point, the algorithm checks if there are enough neighboring points within the distance threshold to form a dense region. If there are, the region is expanded to include the neighboring points, and the process is repeated until no more points can be added. The process then continues with a new arbitrary point that has not yet been assigned to any cluster.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/dbscan.png" alt="DBSCAN" width="500 height="300">


Image Source: [Machine Learning Geek](https://machinelearninggeek.com/dbscan-clustering/)



The working of DBSCAN can be summarized in the following steps:

1. DBSCAN requires two parameters to be set before running the  lgorithm - epsilon (ε) and the minimum number of points required to form a dense region (minPts).
2. The algorithm starts by randomly selecting a point from the dataset.
3. It then checks if there are at least minPts points within a radius of ε around this point.
4. If there are, it forms a dense region around the point, and all the points within the radius are marked as belonging to the same cluster.
5. The algorithm repeats this process for all the points in the dense region until no more points can be added to the cluster.
6. If there are not enough points within ε of the starting point to form a dense region, the point is marked as noise and the algorithm moves on to the next unvisited point.

DBSCAN can handle clusters of different shapes and sizes and can also identify noise points that do not belong to any cluster.

### Mathematical Formulation 

Mathematically, the density of a point p can be defined as the number of points within a radius ε of p. Let D be the dataset containing n points {p1, p2, …, pn}, and let p be a point in D. Then, the density of p can be represented as:

$$ \rho_p = \left|\{q \in D: dist(p,q) \leq \epsilon\}\right| $$

where dist(p,q) represents the distance between points p and q.

A point p is said to be a core point if its density is greater than or equal to minPts. A point q is said to be directly reachable from p if q is within ε distance of p and p is a core point. A point r is said to be reachable from p if there exists a sequence of points p1, p2, ..., pn such that p1 = p, pn = r, and pi+1 is directly reachable from pi for 1 ≤ i < n.

A cluster is formed by a set of core points that are reachable from each other. Any point that is not part of a cluster is marked as noise. The algorithm outputs a set of clusters and a set of noise points.


## References/Resources 
- [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [DBSCAN Clustering - Machine Learning Geek](https://machinelearninggeek.com/dbscan-clustering/)
