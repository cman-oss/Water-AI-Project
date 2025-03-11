import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load dataset using the input_1 variable that contains the file path
data = pd.read_csv('/workspaces/Water-AI-Project/AI/water_potability (2).csv')

# Separate features (X) and target variable (y)
X = data.drop('Potability', axis=1)
y = data['Potability']


# Define categorical and numerical columns
# Looking at the data, all columns appear to be numerical, so we don't have categorical columns
categorical_cols = []  # Empty list since we don't have categorical columns
numerical_cols = X.columns.tolist()  # All columns are numerical

# Create and apply the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ])
X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)
# Update the preprocessor to include SimpleImputer for handling missing values
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', SimpleImputer(strategy='mean'), numerical_cols)
    ])

# Transform the data with the updated preprocessor
X_processed = preprocessor.fit_transform(X)

# Split the processed data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Initialize and train the model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=preprocessor.get_feature_names_out(),  # Get transformed feature names
    filled=True,
    rounded=True,
    max_depth=2  # Limit depth for readability
)
plt.show()

feature_importances = model.feature_importances_
features = preprocessor.get_feature_names_out()

# Create a DataFrame to display importances
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

final_model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
final_model.fit(X_train, y_train)

# Create a dictionary with the input values
input_data = pd.DataFrame({
    'ph': [7.1],
    'Hardness': [2.1],
    'Solids': [0],  # Using 0 as placeholder
    'Chloramines': [0],
    'Sulfate': [0],
    'Conductivity': [0],
    'Organic_carbon': [0],
    'Trihalomethanes': [0],
    'Turbidity': [0]
})

# Transform the input data using the same preprocessor
input_processed = preprocessor.transform(input_data)

# Make prediction using the final model
prediction = final_model.predict(input_processed)

print(f"Predicted potability: {prediction[0]}")