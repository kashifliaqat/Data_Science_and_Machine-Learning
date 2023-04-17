# Unsupervised Learning
Unsupervised learning is a type of machine learning that involves discovering patterns and relationships in data without explicit supervision or guidance from labeled examples. 

In other words, the algorithm is left to find structure and patterns in the data on its own, without being told what features to look for or what the correct output should be. This makes unsupervised learning particularly useful in scenarios where labeled data is not available or difficult to obtain.

**Mathematically**, unsupervised learning can be formulated as finding a representation or model for a given dataset, without any prior knowledge about the underlying structure. The objective is to minimize a certain cost function that captures the discrepancy between the input data and the model. This is typically done by iteratively updating the model parameters until convergence is reached.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/unsupervised.PNG" alt="Unsupervised Learning" width="500" height="300">

Image Source: [NIXUS](https://nixustechnologies.com/unsupervised-machine-learning/)

### Types of Unsupervised Learning
There are several types of unsupervised learning, including:

1. **Clustering:** In clustering, the goal is to group together similar data points into clusters based on some similarity measure. Examples of clustering algorithms include k-means, hierarchical clustering, and DBSCAN.

2. **Dimensionality reduction:** In dimensionality reduction, the goal is to reduce the number of features or dimensions of the input data while preserving as much information as possible. Examples of dimensionality reduction algorithms include PCA, t-SNE, and autoencoders.

3. **Anomaly detection:** In anomaly detection, the goal is to identify data points that are significantly different from the rest of the dataset. Examples of anomaly detection algorithms include one-class SVM and isolation forest.

4. **Association rule learning:** In association rule learning, the goal is to discover relationships between variables in the dataset. Examples of association rule learning algorithms include Apriori and FP-growth.

Overall, unsupervised learning plays a crucial role in many areas of data science, from exploratory data analysis to feature engineering and data preprocessing.

## Unsupervised Learning Models 
The following is the list of unsupervised learning algorithms developed and discussed in this repo. 

1. [K-Means Clustering](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/tree/main/Unsupervised_Learning/1_k_means_clustering)
2. [DBSCAN Clustering](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/tree/main/Unsupervised_Learning/2_DBSCAN)
3. [Viusalizing: K-Means vs DBSCAN](https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/tree/main/Unsupervised_Learning/3_Visualizing_k_means_VS_dbscan)


## References
- [Unsupervised Learning Types, Algorithms and Applications - NIXUS](https://nixustechnologies.com/unsupervised-machine-learning/)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, by Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)