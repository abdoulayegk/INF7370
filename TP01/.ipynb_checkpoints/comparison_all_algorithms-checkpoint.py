# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
import seaborn as sns
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import sklearn

sklearn.set_config(transform_output="pandas")

# import jupyter_black

# jupyter_black.load()


df = pd.read_csv("combine_datasets.csv")
# print(df.columns.tolist())

# print(df.describe()


def feature_selection_and_normalization(df):
    """
    Perform feature selection and normalization on the dataset.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to be normalise and the class label.

    Returns:
    tuple: Processed training and testing datasets (X_train, X_test, y_train, y_test).
    """
    # Remplacement des valeurs infinies par NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Séparation des caractéristiques (X) et de la cible (y)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Création d'un pipeline pour imputer les valeurs manquantes et normaliser les données
    pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median"),
            ),  # Imputation des valeurs manquantes avec la médiane
            ("scale", StandardScaler()),  # Normalisation des données
        ]
    )

    # Application du pipeline aux données d'entraînement et de test
    X_train = pd.DataFrame(
        columns=X_train.columns, data=pipeline.fit_transform(X_train)
    )
    X_test = pd.DataFrame(columns=X_test.columns, data=pipeline.transform(X_test))

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
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

    return {"TP Rate": tp_rate, "FP Rate": fp_rate, "F-mesure": f_measure, "AUC": auc}


def decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test)


def bagging(X_train, y_train, X_test, y_test):
    model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=42)
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test)


def adaboost(X_train, y_train, X_test, y_test):
    model = AdaBoostClassifier(random_state=42)
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test)


def gradient_boosting(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test)


def random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test)


def naive_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test)


for algo, metrics in results.items():
    print(f"{algo}: {metrics}")


def main():

    X_train, X_test, y_train, y_test = feature_selection_and_normalization(df)
    print(X_train.shape, X_test.shape)
    results = {
        "Arbre de décision": decision_tree(X_train, y_train, X_test, y_test),
        "Bagging": bagging(X_train, y_train, X_test, y_test),
        "AdaBoost": adaboost(X_train, y_train, X_test, y_test),
        "Boosting de gradient": gradient_boosting(X_train, y_train, X_test, y_test),
        "Forêts d’arbres aléatoires": random_forest(X_train, y_train, X_test, y_test),
        "Classification bayésienne naïve": naive_bayes(
            X_train, y_train, X_test, y_test
        ),
    }


if __name__ == "__main__":
    main()
