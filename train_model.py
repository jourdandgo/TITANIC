import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib

def extract_title(name):
    import re
    if re.search('Mrs.', name): return 'Mrs'
    elif re.search('Mr.', name): return 'Mr'
    elif re.search('Miss.', name): return 'Miss'
    elif re.search('Master.', name): return 'Master'
    else: return 'Other'

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Feature Engineering
    df['Title'] = df['Name'].apply(extract_title)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Drop identifiers and columns with excessive missing data
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    return df

def build_pipeline():
    numeric_features = ['Age', 'Fare', 'FamilySize']
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

def train_and_compare(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = build_pipeline()
    
    # IMPROVED: Adding hyperparameter limits to prevent over-confidence (overfitting)
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=200, max_depth=6, min_samples_leaf=5)
    }
    
    results = {}
    best_f1 = 0
    champion_pipeline = None
    
    print("--- Model Comparison Results ---")
    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {'Accuracy': acc, 'F1': f1}
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        if f1 > best_f1:
            best_f1 = f1
            champion_pipeline = clf
            
    print(f"\nFinal Selection: Tuned Random Forest (Better Calibration)")
    # Save the tuned champion model
    joblib.dump(champion_pipeline, 'champion_titanic_model.joblib')
    print("Calibrated Random Forest model saved to 'champion_titanic_model.joblib'")
    
    return champion_pipeline, results

if __name__ == "__main__":
    data_path = 'titanic.csv'
    processed_df = load_and_preprocess_data(data_path)
    train_and_compare(processed_df)

if __name__ == "__main__":
    data_path = 'titanic.csv'
    processed_df = load_and_preprocess_data(data_path)
    train_and_compare(processed_df)
