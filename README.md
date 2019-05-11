# Classifying Abalone Data
For this project, whether the number of rings is greater or less than 10 (y=1 for rings > 10 and =0 otherwise) was predicted using machine learning binary classification techniques, namely Fisher's linear discriminant, Logistic Regression and the Shared Covariance Model. The evaluation methods used to compare model fit are classification accuracy, precison-recall, ROC curve, and F1 score. Basis functions, namely quadratic basis functions and radial basis functions, were also used to map raw data into feature space to improve model fit. 

# Dataset 
The data is obtained from UCI Machine Learning Repository - https://archive.ics.uci.edu/ml/datasets/abalone

# Findings
i) Precision-recall curves were more informative than ROC curves. This might be due to class imbalances in both the dataset, with target mean 0.346. As the calculation of precision and recall does not make use of true negatives, the indicator is only concerned with the correct prediction of the minority class, class 1. As ROC does not make this differentiation, it tends to be over-optimistic about model skill when there are class imbalances.

ii) Using quadratic basis functions and radial basis functions improved overall model performance across all three classification models. This suggests that the decision boundaries in dataspace of the dataset is not linear. 

iii) The shared covariance model performs better than logistic regression in terms of precision and recall. 

# Disclaimer
The code supplied in the zip file, fomlads.zip, is obtained for the course material of INST0060 (Foundations of Machine Learning and Data Science), a module offered by University College London, Department of Information Studies. 


