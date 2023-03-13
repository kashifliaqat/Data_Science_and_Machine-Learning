import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        
        # initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        # keep track of loss for each iteration
        self.losses = []
        
        for i in range(self.num_iterations):
            # forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            # compute cost
            cost = -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
            self.losses.append(cost)

            # backward pass
            dz = y_pred - y
            dw = np.dot(X.T, dz) / num_samples
            db = np.mean(dz)

            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred > 0.5).astype(int)


# load data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# normalize data
X = (X - X.mean(axis=0)) / X.std(axis=0)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = LogisticRegression()
model.fit(X_train, y_train)

# calculate accuracy and confusion matrix on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f"Test accuracy: {accuracy:.2f}")
print("Confusion matrix:")
print(conf_mat)

# calculate ROC curve and AUC on test set
from sklearn.metrics import roc_curve, auc
y_pred = model.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

