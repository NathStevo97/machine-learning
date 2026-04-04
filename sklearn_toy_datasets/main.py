import pandas as pd
import shap
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():

    #
    # Data Loading and Exploration
    #

    # Load the wine dataset
    wine_data = load_wine()

    # Convert dataset to a DataFrame for better visualization
    wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)

    # Set Target Label
    wine_df["target"] = wine_data.target
    # Preview Target
    print(f"Target classes: {wine_df['target'].unique()}")

    # Display Preview of the Dataset (first 5 rows)
    print(wine_df.head())

    # Print Dataset Info
    print(wine_df.info())

    # Print Summary Statistics
    print(wine_df.describe())

    # Preview Dataset (Last 5 rows)
    print(wine_df.tail())

    #
    # Preprocessing
    #

    # Separate features and target variable (class)
    X = wine_df[wine_data.feature_names].copy()
    y = wine_df["target"].copy()

    # Log target variable
    print(f"Target variable for prediction: {y.name}")

    # Initialize the StandardScaler to standardize the features with mean=0 and variance=1
    scaler = StandardScaler()

    # Fit the scaler to the features
    scaler.fit(X)

    # Transform the features using the fitted scaler
    X_scaled = scaler.transform(X)

    # Preview first instance of the scaled features
    print(X_scaled[0])

    #
    # Model Training and Evaluation
    #

    # split the dataset into training and testing sets
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X_scaled, y, train_size=0.7, random_state=25
    )

    # Verify Splits
    print(f"Training size: {round(len(X_train_scaled) / len(X) * 100)}%")
    print(f"Testing size: {round(len(X_test_scaled) / len(X) * 100)}%")

    # Initialize the models
    logistic_regression = LogisticRegression()
    svm = SVC()
    decision_tree = DecisionTreeClassifier()

    # Train the models
    lr_model = logistic_regression.fit(X_train_scaled, y_train)
    svm_model = svm.fit(X_train_scaled, y_train)
    decision_tree_model = decision_tree.fit(X_train_scaled, y_train)

    # Generate Predictions with Each Model
    lr_predictions = lr_model.predict(X_test_scaled)
    svm_predictions = svm_model.predict(X_test_scaled)
    dt_predictions = decision_tree_model.predict(X_test_scaled)

    #
    # Evaluate the Models
    #

    model_predictions = {
        "Logistic Regression": lr_predictions,
        "Support Vector Machine": svm_predictions,
        "Decision Tree": dt_predictions,
    }

    for model, preds in model_predictions.items():
        print(f"{model} Results:")
        print(f"\n{classification_report(y_test, preds)}", sep="\n\n")

    explainer = shap.KernelExplainer(lr_model.predict, X_train_scaled, link="logit")
    shap_values = explainer.shap_values(X_test_scaled, nsamples=100)

    shap.summary_plot(shap_values, X_test_scaled, feature_names=wine_data.feature_names)


if __name__ == "__main__":
    main()
