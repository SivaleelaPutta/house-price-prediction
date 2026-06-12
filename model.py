import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

#Step 1: Load dataset
df = pd.read_csv("dataset.csv")
print("Dataset Loaded")
print(df.head())

#Step 2: Features and Target
X = df[["area", "bedrooms", "bathrooms", "floors", "age", "location_score"]]
y = df["price"]

#Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Step 4: Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model Training Completed")

#Step 5: Predictions
y_pred = model.predict(X_test)

print("Sample Predictions:")
print(y_pred[:5])

#Step 6: Accuracy
accuracy = r2_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

#Step 7: Save Model
joblib.dump(model, "house_price_model.pkl")
print("Model Saved Successfully")