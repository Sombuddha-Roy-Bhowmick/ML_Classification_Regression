# ML Classification & Regression
This repository has basic principles and codes for Supervised learning techniques (Classification &amp; Regression) of different Machine Learning Algorithms.

#  Dataset Description:

This synthetic dataset provides detailed information on lung cancer patients, covering demographic attributes, medical history, treatment specifics, and survival outcomes. Although generated synthetically, the dataset closely mirrors real-world scenarios encountered in clinical settings, making it suitable for predictive modeling, prognosis assessment, and treatment efficacy analysis in lung cancer research.

# Dataset Citation: 
Rashad Mammadov. (2024). Lung Cancer Prediction [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/5109540

# Classification Results

Target = "Cancer Stage"
The visualisation of results can be found: "roc_curve.png" 

Classifier: SVM
Accuracy: 0.2499295576218653 
AUC: 0.5088218564880819 
------------------------------ 
Classifier: 
KNN Accuracy: 0.2583826429980276 
AUC: 0.5127997701323501 
------------------------------ 
Classifier: Decision Tree 
Accuracy: 0.24485770639616794 
AUC: 0.48935959099051113 
------------------------------ 
Classifier: Neural Network 
Accuracy: 0.2569737954353339 
AUC: 0.5134307420347073 
------------------------------ 
Classifier: Logistic Regression 
Accuracy: 0.2465483234714004 
AUC: 0.504106131660809 
------------------------------ 
Classifier: Naive Bayes 
Accuracy: 0.24471682163989855 
AUC: 0.5061201935379426 
------------------------------ 
Classifier: LDA 
Accuracy: 0.24570301493378416 
AUC: 0.50417369499409 
------------------------------ 
Classifier: QDA 
Accuracy: 0.25232459847844463 
AUC: 0.5026335012298484 
------------------------------ 


Best Model based on Accuracy: 
Classifier: KNN 
Accuracy: 0.2583826429980276 
AUC: 0.5127997701323501 
------------------------------ 
Best Model based on AUC: 
Classifier: Neural Network 
Accuracy: 0.2569737954353339 
AUC: 0.5134307420347073
------------------------------ 

# Regression Results

Target = "Survival Months"
The visualisation of results can be found: "regression_metrics_plot.jpg"

Linear Regression:
Mean Squared Error = 1167.7170402224003
Mean Absolute Error = 29.502001349417352
R-squared = -0.0026643265962240648
---------------------------------
Ridge Regression:
Mean Squared Error = 1167.7161450362585
Mean Absolute Error = 29.50199511179015
R-squared = -0.0026635579415079658
---------------------------------
Lasso Regression:
Mean Squared Error = 1164.755835850475
Mean Absolute Error = 29.476776900480917
R-squared = -0.00012167809044849598
---------------------------------
Neural Network Regression:
Mean Squared Error = 1497.455945856024
Mean Absolute Error = 32.36436074329154
R-squared = -0.2857957928517345
---------------------------------
Decision Tree Regression:
Mean Squared Error = 2351.590588898281
Mean Absolute Error = 39.53705269089885
R-squared = -1.0192014957653113
---------------------------------
Random Forest Regression:
Mean Squared Error = 1182.9128572132995
Mean Absolute Error = 29.58530994646379
R-squared = -0.01571226807985404
---------------------------------
KNN Regression:
Mean Squared Error = 1390.9410256410254
Mean Absolute Error = 31.407917723302344
R-squared = -0.19433638353329097
---------------------------------
Support Vector Machines (SVM):
Mean Squared Error = 1168.521575176482
Mean Absolute Error = 29.507994878905237
R-squared = -0.0033551433524852126
---------------------------------
Gaussian Process Regression:
Mean Squared Error = 4712.00245592292
Mean Absolute Error = 59.56022306752462
R-squared = -3.045977412890953
---------------------------------
Polynomial Regression:
Mean Squared Error = 1244.7247280630982
Mean Absolute Error = 30.10701054442127
R-squared = -0.06878724748536325
---------------------------------

Best Model: Lasso Regression
Mean Squared Error = 1164.755835850475
Mean Absolute Error = 29.476776900480917
R-squared = -0.00012167809044849598 
---------------------------------

