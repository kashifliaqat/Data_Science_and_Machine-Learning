# Decision Trees

## Tasks
Use decision tree to perform classification to predict whether a patient has diabetes or not.

## Dataset
The [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set) is a collection of medical data from female patients of Pima Indian heritage. The dataset contains information on various medical variables, including glucose level, blood pressure, and body mass index, and their relation to the onset of diabetes in the patients.

The dataset consists of 9 variables:

1. Pregnancies: Number of times pregnant
2. Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. BloodPressure: Diastolic blood pressure (mm Hg)
4. SkinThickness: Triceps skin fold thickness (mm)
5. Insulin: 2-Hour serum insulin (mu U/ml)
6. BMI: Body mass index (weight in kg/(height in m)^2)
7. DiabetesPedigreeFunction: Diabetes pedigree function
8. Age: Age in years
9. Outcome: Class variable (0 or 1)  indicating whether or not the patient developed diabetes within 5 years of the initial examination. A value of 1 indicates that the patient developed diabetes, while a value of 0 indicates that they did not develop diabetes.

## The Decision Tree Algorithm

Decision tree is a supervised learning algorithm used for classification and regression tasks. It is a type of model that learns simple decision rules from the data and builds a tree-like structure to represent the decision-making process.

The idea behind decision tree is to recursively split the data into subsets based on the values of the input features such that the target variable is more homogeneous within each subset. This is done by choosing the feature that maximizes the information gain, which is a measure of the reduction in uncertainty about the target variable after the split. The split is continued until a stopping criterion is met, such as a maximum depth of the tree or a minimum number of samples required to split a node.

The decision tree can be represented as a flowchart-like structure where each internal node represents a test on an input feature, each branch represents the outcome of the test, and each leaf node represents a class label or a regression value. The decision rule at each internal node is simply a threshold on the input feature, and the decision boundary between two classes is a hyperplane perpendicular to the feature axis.

<p align="center"><img src="https://github.com/kashifliaqat/Data_Science_and_Machine-Learning/raw/main/Images/decision_tree.png" alt="Decision Tree" width="500" height="300">

Source for Images: [DEVOPS](https://www.devops.ae/decision-tree-classification-algorithm/)

### Mathematical Formulation

The decision tree algorithm can be formulated as follows:

Let D be the training set of N samples {(x1, y1), (x2, y2), ..., (xN, yN)}, where xi is the vector of input features and yi is the target variable.

Let F be the set of input features {f1, f2, ..., fd}.

Let t be the decision tree model, represented as a tree-like structure with internal nodes, branches, and leaf nodes.

The algorithm starts with the root node of the tree, which contains all the samples in D.

At each internal node, the algorithm selects the feature fi that maximizes the information gain, defined as:

$$Gain(D, f_i) = Entropy(D) - \sum_{j=1}^{k} \frac{|D_j|}{|D|} * Entropy(D_j)$$

where Dj is the subset of samples in D that have the value vj of feature fi, and Entropy(D) is the entropy of the target variable in D, defined as:

$$Entropy(D) = - \sum_{i=1}^{k} p_i * log_2(p_i)$$

where pi is the proportion of samples in D that belong to class i.

The algorithm creates a child node for each possible value of fi, and splits the samples in D into the corresponding subsets.

The algorithm recursively applies the same process to each child node until a stopping criterion is met, such as a maximum depth of the tree or a minimum number of samples required to split a node.

At each leaf node, the algorithm assigns the most frequent class label in the subset of samples as the predicted class label for any new input that falls in that region.

Overall, decision tree is a powerful and interpretable algorithm that can handle both categorical and numerical input features and can capture complex nonlinear relationships between the input features and the target variable. However, it is prone to overfitting when the tree is too deep or when the training set is too small, and it may fail to capture interactions between different input features. Therefore, it is often used in ensemble methods such as random forests or boosting to improve its performance and robustness.


## Resources/References
1. [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
2. [Decision Tree Classification Algorithm - DEVOPS](https://www.devops.ae/decision-tree-classification-algorithm/)
