import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

custom_colors = ["#7400ff", "#a788e4", "#d216d2", "#ffb500", "#36c9dd"]


def create_balanced_subset(data_path, output_path, polluter_ratio=0.05):
    """
    Crée un sous-ensemble de données où les pollueurs représentent
    0.05 pourcentage  des utilisateurs légitimes

    Args:
        data_path (str): Chemin vers le fichier de données original
        output_path (str): Chemin pour sauvegarder le nouveau sous-ensemble
        polluter_ratio (float): Ratio désiré de pollueurs par rapport aux utilisateurs légitimes
    """
    # Chargement des données
    df = pd.read_csv(data_path)

    # Séparation des utilisateurs légitimes et des pollueurs
    legitimate_users = df[df["Class"] == 0]
    polluters = df[df["Class"] == 1]

    # Calcul du nombre de pollueurs à conserver
    n_legitimate = len(legitimate_users)
    n_polluters_needed = int(n_legitimate * polluter_ratio)

    # Sélection aléatoire des pollueurs
    selected_polluters = polluters.sample(n=n_polluters_needed, random_state=42)

    # Création du nouveau dataset
    balanced_df = pd.concat([legitimate_users, selected_polluters])

    # Mélange des données
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Sauvegarde du nouveau dataset
    balanced_df.to_csv(output_path, index=False)

    # Affichage des statistiques
    print(f"Nombre d'utilisateurs légitimes : {len(legitimate_users)}")
    print(f"Nombre de pollueurs : {len(selected_polluters)}")
    print(
        f"Ratio pollueurs/légitimes : {len(selected_polluters)/len(legitimate_users):.3f}"
    )


def feature_selection_and_normalization(df):
    """
    Perform feature selection and normalization on the dataset.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to be normalized and the class label.

    Returns:
    tuple: Processed training and testing datasets (X_train, X_test, y_train, y_test).
    """
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Split features (X) and target (y)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create pipeline for imputing missing values and normalizing data
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    # Apply pipeline to training and test data
    X_train = pd.DataFrame(
        columns=X_train.columns, data=pipeline.fit_transform(X_train)
    )
    X_test = pd.DataFrame(columns=X_test.columns, data=pipeline.transform(X_test))

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using various metrics.

    Parameters:
    model: Trained classifier model
    X_test (pd.DataFrame): Test features
    y_test (pd.Series): Test labels

    Returns:
    dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tp_rate = tp / (tp + fn)
    fp_rate = fp / (fp + tn)
    f_measure = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    return {
        "TP Rate": tp_rate,
        "FP Rate": fp_rate,
        "F-mesure": f_measure,
        "AUC - ROC Score": auc,
    }


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple classification models, and plot ROC curves.

    Parameters:
    X_train, X_test, y_train, y_test: Training and test datasets

    Returns:
    dict: Dictionary containing results for each model
    """
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Bagging": BaggingClassifier(),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    results = {}
    roc_auc_scores = {}

    # Plot the ROC curves
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Predict the probabilities
        y_probs = model.predict_proba(X_test)[:, 1]

        # Calculate the AUC - ROC score
        roc_auc = roc_auc_score(y_test, y_probs)
        roc_auc_scores[name] = roc_auc

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_probs)

        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

        # Evaluate the model and store the results
        results[name] = evaluate_model(model, X_test, y_test)

    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], color=custom_colors[2], lw=2, linestyle="--")

    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        "ROC Curve Comparison for imbalanced_data with 0.05 % of legitimate users"
    )
    plt.legend(loc="lower right")
    plt.show()

    # Print the AUC - ROC scores for each model
    for name, score in roc_auc_scores.items():
        print(f"{name}: AUC - ROC = {score:.3f}")

    return results


def display_results(results):
    """
    Display evaluation results for all models.

    Parameters:
    results (dict): Dictionary containing results for each model
    """
    for model_name, metrics in results.items():
        print(f"\n======================={model_name}:====================")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")


def main():
    """Main function to orchestrate the model evaluation process."""
    # Load data
    df = pd.read_csv("imbalanced_data.csv")

    # Setup visualization style
    # setup_visualization_style()

    # Preprocess data
    X_train, X_test, y_train, y_test = feature_selection_and_normalization(df)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Display results
    display_results(results)


if __name__ == "__main__":
    # Remplacer ces chemins par vos chemins réels
    input_path = "combine_datasets.csv"
    output_path = "imbalanced_data.csv"

    create_balanced_subset(input_path, output_path)
    main()
