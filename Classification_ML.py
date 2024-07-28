import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('lung_cancer_data.csv')


categorical_columns = ['Gender', 'Smoking_History', 'Tumor_Location', 'Stage', 'Treatment', 'Ethnicity', 'Insurance_Type', 'Family_History', 'Comorbidity_Diabetes', 'Comorbidity_Hypertension', 'Comorbidity_Heart_Disease', 'Comorbidity_Chronic_Lung_Disease', 'Comorbidity_Kidney_Disease', 'Comorbidity_Autoimmune_Disease', 'Comorbidity_Other']


if 'Stage' not in data.columns:
    raise ValueError("The 'Stage' column is not present in the dataset.")


target = 'Stage'
label_encoder = LabelEncoder()
data[target] = label_encoder.fit_transform(data[target])


features = [col for col in data.columns if col != target]

X = data[features]
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


numerical_columns = [col for col in X.columns if col not in categorical_columns + ['Patient_ID']]
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, [col for col in categorical_columns if col in X.columns])
    ])


classifiers = {
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Neural Network': MLPClassifier(max_iter=1000),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis()
}


results = {}

for name, clf in classifiers.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else pipeline.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=y_test.unique()[0])
    roc_auc = auc(fpr, tpr)

    results[name] = {
        'accuracy': acc,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'model': pipeline
    }

for name, result in results.items():
    print(f"Classifier: {name}")
    print(f"Accuracy: {result['accuracy']}")
    print(f"AUC: {result['roc_auc']}")
    print("-" * 30)

best_model_acc = max(results, key=lambda x: results[x]['accuracy'])
best_model_auc = max(results, key=lambda x: results[x]['roc_auc'])

print("\nBest Model based on Accuracy:")
print(f"Classifier: {best_model_acc}")
print(f"Accuracy: {results[best_model_acc]['accuracy']}")
print(f"AUC: {results[best_model_acc]['roc_auc']}")

print("\nBest Model based on AUC:")
print(f"Classifier: {best_model_auc}")
print(f"Accuracy: {results[best_model_auc]['accuracy']}")
print(f"AUC: {results[best_model_auc]['roc_auc']}")


plt.figure(figsize=(10, 8))
for name, result in results.items():
    if len(result['fpr']) > 0 and len(result['tpr']) > 0:  
        plt.plot(result['fpr'], result['tpr'], label=f"{name} (AUC = {result['roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

