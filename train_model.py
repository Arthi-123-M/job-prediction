import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Create training directory if it doesn't exist
os.makedirs('training', exist_ok=True)

def create_sample_data():
    """Create sample job posting data"""
    np.random.seed(42)
    n_samples = 1000
    
    # Features: [has_salary, has_company_logo, has_questions, 
    #            title_length, description_length, requirements_length,
    #            has_telecommuting, has_healthcare, has_401k, has_bonus]
    
    data = {
        'has_salary': np.random.randint(0, 2, n_samples),
        'has_company_logo': np.random.randint(0, 2, n_samples),
        'has_questions': np.random.randint(0, 2, n_samples),
        'title_length': np.random.randint(5, 100, n_samples),
        'description_length': np.random.randint(50, 2000, n_samples),
        'requirements_length': np.random.randint(20, 500, n_samples),
        'has_telecommuting': np.random.randint(0, 2, n_samples),
        'has_healthcare': np.random.randint(0, 2, n_samples),
        'has_401k': np.random.randint(0, 2, n_samples),
        'has_bonus': np.random.randint(0, 2, n_samples),
    }
    
    # Create target (0 = Fake, 1 = Real)
    # Fake jobs tend to have: no salary, no logo, no benefits, etc.
    df = pd.DataFrame(data)
    
    # Generate target based on features (for demo purposes)
    y = (
        df['has_salary'] * 0.2 +
        df['has_company_logo'] * 0.15 +
        df['has_questions'] * 0.1 +
        df['has_telecommuting'] * 0.1 +
        df['has_healthcare'] * 0.15 +
        df['has_401k'] * 0.1 +
        df['has_bonus'] * 0.1 +
        (df['description_length'] > 200) * 0.1
    ) > 0.4  # Threshold for real job
    
    y = y.astype(int)
    
    return df, y

def train_models():
    print("Loading and preparing data...")
    
    # Use sample data (replace with your actual dataset)
    X, y = create_sample_data()
    
    # Save feature names for later use
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(X_train, y_train)
    logistic_acc = logistic_model.score(X_test, y_test)
    print(f"Logistic Regression Accuracy: {logistic_acc:.2%}")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_acc = rf_model.score(X_test, y_test)
    print(f"Random Forest Accuracy: {rf_acc:.2%}")
    
    # Save models in the main directory
    print("\nSaving models...")
    joblib.dump(logistic_model, '../logistic_model.pkl')
    joblib.dump(rf_model, '../rf_model.pkl')
    joblib.dump(feature_names, '../feature_names.pkl')
    
    print("Models saved successfully!")
    print("- logistic_model.pkl")
    print("- rf_model.pkl")
    print("- feature_names.pkl")
    
    return logistic_model, rf_model, feature_names

if __name__ == "__main__":
    # Change to training directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train_models()
