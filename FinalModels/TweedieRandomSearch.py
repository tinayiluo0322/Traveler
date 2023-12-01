# import libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option("mode.copy_on_write", True)

# read in the dataset
df = pd.read_csv("../data/InsNova_data_2023_train.csv")

# DATA PREPROCESSING
# modify the data type of the high_education_ind column
df["high_education_ind"] = df["high_education_ind"].astype("object")
# Identify categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
# Perform one-hot encoding for categorical variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate predictors and target
X = df.drop(columns=["id", "clm", "numclaims", "claimcst0"], axis=1)
y = df["claimcst0"]
# Split the data into training and testing sets
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)


# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# MODELING
model = linear_model.TweedieRegressor()
# Define the parameter distribution for random search
param_dist = {
    "power": np.arange(0, 2.0, 0.01),  # Adjust as needed
    "alpha": np.arange(0, 1.0, 0.01),  # Adjust as needed
    "max_iter": np.arange(10000, 50000, 1000),
    "solver": ["lbfgs", "newton-cholesky"],
    "warm_start": [True, False],
}

# Create the random search with cross-validation
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42,
)

# Fit the grid search to the data
random_search.fit(X_train_scaled, y_train)

# Print the best parameters and corresponding score
print("Best Parameters: ", random_search.best_params_)
print("Best Negative Mean Squared Error: ", random_search.best_score_)

# TESTING
# Test Prediction
# read in dataset
test = pd.read_csv("../data/InsNova_data_2023_vh.csv")


# DATA PREPROCESSING
test["high_education_ind"] = test["high_education_ind"].astype("object")
# Identify categorical columns
categorical_cols = test.select_dtypes(include=["object"]).columns
# Perform one-hot encoding for categorical variables
test = pd.get_dummies(test, columns=categorical_cols, drop_first=True)
test = test.rename(columns={"high_education_ind_1": "high_education_ind_1.0"})
# Separate predictors and target
X_test = test.drop(columns=["id"], axis=1)
X_test_scaled = scaler.transform(X_test)

# PREDICTION
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
# Clip the predictions to be non-negative
y_pred = np.clip(y_pred, a_min=0, a_max=None)

# assign the prediction to the test dataset
test["Predict"] = y_pred
test["Predict"].describe()

# creat submission file
submission = test[["id", "Predict"]]
submission.to_csv("../output/Tweedie_random_submission.csv", index=False)
