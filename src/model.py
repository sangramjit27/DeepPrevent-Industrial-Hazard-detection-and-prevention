import numpy as np
import pandas as pd
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\hp\\Desktop\\IOCL-AOD, Digboi\\iocl_machinery_dataset.csv")
df.head()
# Features (X) and Target (y)
X = df[['temperature', 'vibration', 'oil_flow', 'rpm']].values
y = df['RUL'].values

# Save original mean & std for later (for Streamlit)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
class CustomLinearRegressor:
    def __init__(self, learning_rate=0.01, epochs=2000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.loss_history = []
        self.X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias
        self.y = y.reshape(-1, 1)
        self.theta = np.random.randn(self.X.shape[1], 1)

        start = time.time()
        for epoch in range(self.epochs):
            gradients = 2 / self.X.shape[0] * self.X.T.dot(self.X.dot(self.theta) - self.y)
            self.theta -= self.learning_rate * gradients
            loss = np.mean((self.X.dot(self.theta) - self.y)**2)
            self.loss_history.append(loss)
        end = time.time()
        self.training_time = end - start

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta).flatten()

# Train Custom Model
custom_model = CustomLinearRegressor(learning_rate=0.01, epochs=2000)
custom_model.fit(X_train, y_train)
y_pred_custom = custom_model.predict(X_test)

# Metrics
custom_r2 = r2_score(y_test, y_pred_custom)
custom_mae = mean_absolute_error(y_test, y_pred_custom)
custom_rmse = np.sqrt(mean_squared_error(y_test, y_pred_custom))

print(f"Custom Model R²: {custom_r2:.4f}, MAE: {custom_mae:.2f}, RMSE: {custom_rmse:.2f}")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

print(f"Ridge R²: {ridge_r2:.4f}, MAE: {ridge_mae:.2f}, RMSE: {ridge_rmse:.2f}")
rf = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_r2 = r2_score(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"Random Forest R²: {rf_r2:.4f}, MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")
gb = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

gb_r2 = r2_score(y_test, y_pred_gb)
gb_mae = mean_absolute_error(y_test, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))

print(f"Gradient Boosting R²: {gb_r2:.4f}, MAE: {gb_mae:.2f}, RMSE: {gb_rmse:.2f}")
# Save Custom Model
joblib.dump(custom_model, "trained_model.pkl")

# Save Sklearn Models
joblib.dump(ridge, "ridge_model.pkl")
joblib.dump(rf, "rf_model.pkl")
joblib.dump(gb, "gb_model.pkl")

# Save Normalization Stats
stats_df = pd.DataFrame({"mean": X_mean, "std": X_std})
stats_df.to_csv("scaling_stats.csv", index=False)

print(" All models & scaling stats saved successfully!")
metrics_df = pd.DataFrame([
    {"Model": "Custom Linear Regression", "R2": custom_r2, "MAE": custom_mae, "RMSE": custom_rmse},
    {"Model": "Ridge Regression", "R2": ridge_r2, "MAE": ridge_mae, "RMSE": ridge_rmse},
    {"Model": "Random Forest", "R2": rf_r2, "MAE": rf_mae, "RMSE": rf_rmse},
    {"Model": "Gradient Boosting", "R2": gb_r2, "MAE": gb_mae, "RMSE": gb_rmse}
])

metrics_df.to_csv("model_performance.csv", index=False)
print(" Model performance saved to model_performance.csv")
print(metrics_df)
