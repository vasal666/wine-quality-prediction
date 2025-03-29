import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("🚀 Starting wine quality prediction training...")

# Load data
try:
    df = pd.read_csv("winequality-red.csv", sep=";")
    print(f"✅ Loaded data with {len(df)} rows")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    exit()

# Preprocess
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
X = df.drop('quality', axis=1)
y = df['quality']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, 'wine_quality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("💾 Saved model and scaler")

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"📊 Model accuracy: {accuracy:.2f}")
print("🎉 Training complete!")