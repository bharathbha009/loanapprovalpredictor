
# train_model.py
# Re-train the model from the CSV dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv('loan_dataset.csv')
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

categorical_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
], remainder='passthrough')

pipeline = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X, y)
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print('Model trained and saved to loan_model.pkl')
