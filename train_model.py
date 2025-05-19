import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import json
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Configuration
RED_WINE_PATH = "winequality-red.csv"
WHITE_WINE_PATH = "winequality-white.csv"
MODEL_FILE = 'wine_quality_model.pkl'
SCALER_FILE = 'scaler.pkl'
FEATURES_FILE = 'features.json'
RANDOM_STATE = 42

def load_and_merge_data():
    """Load and merge red/white wine datasets with feature engineering"""
    red = pd.read_csv(RED_WINE_PATH, sep=";")
    white = pd.read_csv(WHITE_WINE_PATH, sep=";")
    
    # Add wine type
    red['wine_type'] = 0  # 0 for red
    white['wine_type'] = 1  # 1 for white
    
    # Combine datasets
    df = pd.concat([red, white], ignore_index=True)
    
    # Feature engineering
    df['acid_balance'] = df['citric acid'] / (df['volatile acidity'] + 1e-6)
    df['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-6)
    df['alcohol_to_acid'] = df['alcohol'] / (df['fixed acidity'] + 1e-6)
    
    return df

def preprocess_data(df):
    """Handle class imbalance safely"""
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Check class distribution
    class_counts = y.value_counts()
    print("\n📊 Original Class Distribution:")
    print(class_counts.sort_index())
    
    # Only apply SMOTE to classes with >5 samples
    min_samples = 6  # SMOTE requires min n_neighbors+1 samples
    valid_classes = class_counts[class_counts >= min_samples].index
    
    if len(valid_classes) < len(class_counts):
        print(f"⚠️ Removing rare classes: {set(class_counts.index) - set(valid_classes)}")
        mask = y.isin(valid_classes)
        X_filtered, y_filtered = X[mask], y[mask]
    else:
        X_filtered, y_filtered = X, y
    
    # Apply SMOTE only if we have valid classes
    if len(valid_classes) >= 2:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_filtered, y_filtered)
        print("\n📈 Resampled Class Distribution:")
        print(pd.Series(y_res).value_counts().sort_index())
    else:
        print("❌ Not enough classes for SMOTE - using original data")
        X_res, y_res = X_filtered, y_filtered
    
    # Fix label encoding for XGBoost
    encoder = LabelEncoder()
    y_res = encoder.fit_transform(y_res)
    
    return train_test_split(X_res, y_res, test_size=0.2, random_state=RANDOM_STATE)

def train_best_model(X_train, y_train):
    """Train optimized model with hyperparameter tuning"""
    models = {
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss', use_label_encoder=False),
        'LightGBM': LGBMClassifier(random_state=RANDOM_STATE)
    }
    
    params = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6]
        },
        'LightGBM': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50]
        }
    }
    
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        print(f"\n🔍 Tuning {name}...")
        grid = GridSearchCV(
            model, 
            params[name], 
            cv=5, 
            scoring='balanced_accuracy',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = grid.best_estimator_
            print(f"⭐ New best: {name} (Balanced Accuracy: {best_score:.3f})")
            print(f"Best params: {grid.best_params_}")
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n📈 Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='balanced_accuracy')
    print("\n🎯 Cross-Validation Scores:", cv_scores)
    print("Mean Balanced Accuracy: {:.2f} ± {:.2f}".format(cv_scores.mean(), cv_scores.std()))

def save_artifacts(model, scaler, features):
    """Save all necessary artifacts"""
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    # Save feature names and engineered features
    with open(FEATURES_FILE, 'w') as f:
        json.dump({
            'original_features': [f for f in features if f not in ['acid_balance', 'sulfur_ratio', 'alcohol_to_acid', 'wine_type']],
            'engineered_features': ['acid_balance', 'sulfur_ratio', 'alcohol_to_acid', 'wine_type']
        }, f, indent=4)
    
    print("\n💾 Saved artifacts:")
    print(f"- Model: {MODEL_FILE}")
    print(f"- Scaler: {SCALER_FILE}")
    print(f"- Features: {FEATURES_FILE}")

def main():
    print("🚀 Starting enhanced wine quality training...")
    
    # Step 1: Data loading and merging
    print("\n📂 Loading datasets...")
    df = load_and_merge_data()
    print(f"✅ Loaded {len(df)} samples (Red: {sum(df['wine_type']==0)}, White: {sum(df['wine_type']==1)})")
    
    # Step 2: Preprocessing
    print("\n⚙️ Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Step 3: Feature scaling
    print("\n🔢 Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Step 4: Model training
    print("\n🤖 Training model...")
    model = train_best_model(X_train, y_train)
    
    # Step 5: Evaluation
    print("\n🧪 Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Step 6: Save artifacts
    print("\n💾 Saving artifacts...")
    save_artifacts(model, scaler, list(df.drop('quality', axis=1).columns))
    
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()
