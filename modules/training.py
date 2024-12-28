import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import streamlit as st

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay
)

# Directory to save models
MODEL_SAVE_DIR = "saved_models"
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

def save_model(model, model_name):
    """
    Save the trained model to a file using joblib.
    
    Args:
        model: Trained model.
        model_name: Name of the model to save (e.g., "svm", "knn").
    """
    model_filename = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{int(time.time())}.pkl")
    joblib.dump(model, model_filename)
    return model_filename

def load_model(model_filename):
    """
    Load a saved model from a file.
    
    Args:
        model_filename: Path to the saved model file.
    
    Returns:
        Loaded model.
    """
    return joblib.load(model_filename)

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    model_filename = save_model(model, "linear_regression")
    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    model_filename = save_model(model, "logistic_regression")
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    model_filename = save_model(model, "decision_tree")
    return model

def train_svm(X_train, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    model_filename = save_model(model, "svm")
    return model

def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    model_filename = save_model(model, "naive_bayes")
    return model

def train_knn(X_train, y_train):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    model_filename = save_model(model, "knn")
    return model

def train_kmeans(X_train, n_clusters=3):
    model = KMeans(n_clusters=n_clusters)
    model.fit(X_train)
    model_filename = save_model(model, "kmeans")
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    model_filename = save_model(model, "random_forest")
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    model_filename = save_model(model, "gradient_boosting")
    return model

def train_adaboost(X_train, y_train):
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    model_filename = save_model(model, "adaboost")
    return model

# Load the model from session state (if available)
def load_trained_model(model_name):
    """
    Load a trained model from the session state or from the saved models directory.
    
    Args:
        model_name: Name of the model to load.
    
    Returns:
        Loaded model.
    """
    if "trained_models" in st.session_state and model_name in st.session_state["trained_models"]:
        return st.session_state["trained_models"][model_name]
    else:
        # If model is not in session state, load from saved models directory
        model_files = [f for f in os.listdir(MODEL_SAVE_DIR) if model_name in f]
        if model_files:
            return load_model(os.path.join(MODEL_SAVE_DIR, model_files[-1]))  # Load the latest model
        else:
            return None

def evaluate_regression(model, X_test, y_test):
    """
    Evaluate a regression model on test data using all available metrics.
    
    Args:
        model: Trained regression model.
        X_test: Test features.
        y_test: Test target values.

    Returns:
        results: Dictionary of calculated metrics.
    """
    y_pred = model.predict(X_test)
    results = {}

    # Calculate all available regression metrics
    results["Mean Squared Error (MSE)"] = mean_squared_error(y_test, y_pred)
    results["Mean Absolute Error (MAE)"] = mean_absolute_error(y_test, y_pred)
    results["R-squared (R²)"] = r2_score(y_test, y_pred)

    return results

def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    results = {}
    visualizations = []

    # Calculate all available classification metrics
    results["Accuracy"] = accuracy_score(y_test, y_pred)
    results["Precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    results["Recall"] = recall_score(y_test, y_pred, average="weighted")
    results["F1-Score"] = f1_score(y_test, y_pred, average="weighted")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    visualizations.append(("Confusion Matrix", cm))

    # ROC-AUC for binary or multi-class classification
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        
        # For multi-class classification, binarize the labels
        n_classes = y_proba.shape[1]  # Get the number of classes
        
        if n_classes == 2:  # Binary classification
            results["ROC-AUC"] = roc_auc_score(y_test, y_proba[:, 1])  # Only use the positive class probabilities
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])  # For binary classification, use positive class
            roc_auc = auc(fpr, tpr)
            visualizations.append(("ROC Curve", (fpr, tpr, roc_auc)))
        else:  # Multi-class classification
            # Binarize the labels for multi-class ROC AUC
            y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
            results["ROC-AUC"] = roc_auc_score(y_test_bin, y_proba, multi_class='ovr', average='weighted')
            
            # Compute ROC curve for each class
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])  # Compare to each class
                roc_auc = auc(fpr, tpr)
                visualizations.append((f"ROC Curve for Class {i}", (fpr, tpr, roc_auc)))

    return results, visualizations

def plot_roc_curve(y_test, y_proba, n_classes):
    """
    Plot an ROC curve for multi-class classification.
    
    Args:
        y_test: True target values.
        y_proba: Predicted probabilities.
        n_classes: Number of classes in the classification problem.
    """
    # Binarize the labels for multi-class ROC
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")

    return fig

def plot_confusion_matrix(cm):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix.
    """
    # Create a new figure and axis
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="viridis", ax=ax)  # Pass the axis to the plot
    ax.set_title("Confusion Matrix")
    
    return fig