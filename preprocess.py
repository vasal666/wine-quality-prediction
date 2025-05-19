import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

print("🔧 Preprocessing data...")

df = pd.read_csv("winequality-red.csv", sep=";")
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
df.to_csv('processed_wine.csv', index=False)

scaler = StandardScaler()
scaler.fit(df.drop('quality', axis=1))
joblib.dump(scaler, 'scaler.pkl')

print("✅ Preprocessing complete!")
print("Saved: processed_wine.csv and scaler.pkl")