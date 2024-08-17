# Data Science Pratical Application 3:
# Comparing Classifiers

# Data:
The dataset comes from the UCI Machine Learning repository. The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns. Our goal is to compare different classifiers, find the best classification model and predict what drives customers to buy bank products. 

# Data Cleanup
This dataset has 41188 rows and 21 columns but there are 12718 unknown strings in 6 columns. It is also an imbalanced dataset because there are 93% in class 0 and 7% in class 1. We have to address all these issues before starting data analytics.

# Data Analytics
We build and compare four classifier models. By using GridSearchCV and turing hyper parameters, we can find the best model. After the fitted model is avaiable, we analyze permutation importances and model coefficients. The result tells us the list of key features that customers like to purchase bank products.

# Findings
- The top three important features are 'cons.price.idx', 'duration' and 'nr.employed'. That means the monthly consumer price index, previous phone call duration and quarterly employed index are positively correlated to the bank marketing campaign accuracy 
- After hyper parameters tuning, SupportVectorMachine is the best model with nonlinear kernel  but it is very slow to train
- LogisticRegression is the best model with default parameters. It is fast to train and we get the feature coefficients 
- KNearestNeighbors is the lowest performance classifier because it doesn't handle imbalanced dataset. However, it is the festest to fit and train the model
- The hyper parameter 'class_weight' plays a very important role to improve model performance because the dataset is highly imbalanced. It works for SupportVectorMachine, LogisticRegression and DecisionTree classifiers.

# Summary
Supervised learning is one of the most useful machine learning technology to solve binary classification and mutli-class classification problems. We analyzed a Portuguese bank marketing campaign by building and comparing four classifier models: LogisticRegression, KNearestNeighbors, DecisionTree and SupportVectorMachine. SupportVectorMachine is the best classifiers among all because it achieves higher scoring with non-linear kernels. However, it uses a lot of CPU power and can be very slow when the dataset is very large. We can speed up the traing time by using powerful GPU or applying PCA dimension reduction techniques.