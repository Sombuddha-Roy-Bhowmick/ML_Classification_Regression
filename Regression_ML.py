import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('lung_cancer_data.csv')

categorical_columns = ['Gender', 'Smoking_History', 'Tumor_Location', 'Stage', 'Treatment', 'Ethnicity', 'Insurance_Type', 'Family_History', 'Comorbidity_Diabetes', 'Comorbidity_Hypertension', 'Comorbidity_Heart_Disease', 'Comorbidity_Chronic_Lung_Disease', 'Comorbidity_Kidney_Disease', 'Comorbidity_Autoimmune_Disease', 'Comorbidity_Other']

numerical_columns = [col for col in data.columns if col not in categorical_columns + ['Survival_Months', 'Patient_ID']]

X = data.drop(columns=['Survival_Months', 'Patient_ID'])

y = data['Survival_Months']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ])


def evaluate_model(model):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Neural Network Regression': MLPRegressor(max_iter=1000),
    'Decision Tree Regression': DecisionTreeRegressor(),
    'Random Forest Regression': RandomForestRegressor(),
    'KNN Regression': KNeighborsRegressor(),
    'Support Vector Machines (SVM)': SVR(),
    'Gaussian Process Regression': GaussianProcessRegressor(),
    'Polynomial Regression': Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])
}

results = {}

for name, model in models.items():
    mse, mae, r2 = evaluate_model(model)
    results[name] =  {
        'Mean Squared Error': mse,
        'Mean Absolute Error': mae,
        'R-squared': r2
    }

    print(f'{name}:')
    print(f'Mean Squared Error = {mse}')
    print(f'Mean Absolute Error = {mae}')
    print(f'R-squared = {r2}')
    print('---------------------------------')


best_model_name = min(results, key=lambda x: results[x]['Mean Squared Error'])
best_model_metrics = results[best_model_name]

print(f'\nBest Model: {best_model_name}')
print(f'Mean Squared Error = {best_model_metrics["Mean Squared Error"]}')
print(f'Mean Absolute Error = {best_model_metrics["Mean Absolute Error"]}')
print(f'R-squared = {best_model_metrics["R-squared"]}')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))


ax1.bar(results.keys(), [result['Mean Squared Error'] for result in results.values()], color='skyblue')
ax1.set_xlabel('Models')
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('Mean Squared Error for Different Regression Models')
ax1.tick_params(axis='x', rotation=90)


ax2.bar(results.keys(), [result['Mean Absolute Error'] for result in results.values()], color='salmon')
ax2.set_xlabel('Models')
ax2.set_ylabel('Mean Absolute Error')
ax2.set_title('Mean Absolute Error for Different Regression Models')
ax2.tick_params(axis='x', rotation=90)


ax3.bar(results.keys(), [result['R-squared'] for result in results.values()], color='lightgreen')
ax3.set_xlabel('Models')
ax3.set_ylabel('R-squared')
ax3.set_title('R-squared for Different Regression Models')
ax3.tick_params(axis='x', rotation=90)

plt.tight_layout()


plt.savefig('regression_metrics_plot.jpg', dpi=600)  

