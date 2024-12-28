import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from modules.dataset import save_uploaded_file
from modules.eda import plot_data_distribution, plot_histogram, plot_boxplot, plot_corr_heatmap, plot_pairplot
from modules.preprocessing import handle_missing_values, encode_categorical, scale_features
from modules.training import (
    train_linear_regression, train_logistic_regression, train_decision_tree, train_svm, 
    train_naive_bayes, train_knn, train_random_forest, train_gradient_boosting, train_adaboost, 
    save_model, load_model, load_trained_model, evaluate_regression, evaluate_classification, 
    plot_confusion_matrix, plot_roc_curve
)
import shutil

# Initialize session state for dataset
if "original_data" not in st.session_state:
    st.session_state["original_data"] = None  # Original dataset
if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = None  # Preprocessed dataset

# Title and Sidebar Navigation
st.title("AutoML Pipeline Builder")
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Select a task:", 
    ["Dataset", "EDA", "Preprocessing", "Training and Evaluation", "Model Comparison", "Model Export"]
)

# Dataset Section
if options == "Dataset":
    st.header("Dataset")

    # File Upload
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # Save the uploaded file to the 'data' folder
        file_path = save_uploaded_file(uploaded_file)
        
        # Load the dataset into the session state
        if uploaded_file.name.endswith('.csv'):
            st.session_state["original_data"] = pd.read_csv(file_path)
        else:
            st.session_state["original_data"] = pd.read_excel(file_path)
        
        # Reset all states
        st.session_state["processed_data"] = None
        st.session_state["trained_models"] = {}
        st.session_state["encoded_data"] = None
        st.session_state["missing_values_handled_data"] = None
        st.session_state["X_train"] = None
        st.session_state["X_test"] = None
        st.session_state["y_train"] = None
        st.session_state["y_test"] = None
        st.success("Dataset uploaded and all states have been reset!")

    # Display dataset information if available
    if st.session_state["original_data"] is not None:
        original_df = st.session_state["original_data"]
        st.write("### Original Dataset Preview")
        st.write(original_df.head())
        st.write("### Basic Information")
        st.write(f"**Shape:** {original_df.shape}")
        st.write(f"**Columns:** {list(original_df.columns)}")
        st.write(f"**Numerical Columns:** {list(original_df.select_dtypes(include=['number']).columns)}")
        st.write(f"**Categorical Columns:** {list(original_df.select_dtypes(include=['object']).columns)}")
    else:
        st.info("Please upload a dataset to proceed.")

# EDA Section
elif options == "EDA":
    st.header("Exploratory Data Analysis")

    # Check if original data is available
    if st.session_state["original_data"] is not None:
        original_df = st.session_state["original_data"]
        
        # Display Original Dataset Preview
        st.write("### Dataset Preview")
        st.write(original_df.head())

        # EDA Options
        eda_option = st.selectbox(
            "Select an EDA technique:",
            ["Summary Statistics", "Data Distribution", "Histogram", "Boxplot", "Correlation Heatmap", "Pairplot"]
        )
        # Perform selected EDA
        if eda_option == "Summary Statistics":
            st.write("### Summary Statistics")
            st.write(original_df.describe(include='all').transpose())
        elif eda_option == "Data Distribution":
            if st.button("Data Distribution"):
                plot_data_distribution(original_df)
        elif eda_option == "Histogram":
            column = st.selectbox("Select a column for histogram:", original_df.select_dtypes(include=['number']).columns)
            if st.button("Plot Histogram"):
                plot_histogram(original_df, column)
        elif eda_option == "Boxplot":
            column = st.selectbox("Select a column for boxplot:", original_df.select_dtypes(include=['number']).columns)
            if st.button("Plot Boxplot"):
                plot_boxplot(original_df, column)
        elif eda_option == "Correlation Heatmap":
            if st.button("Plot Correlation Heatmap"):
                plot_corr_heatmap(original_df)
        elif eda_option == "Pairplot":
            if st.button("Plot Pairplot"):
                plot_pairplot(original_df)
    else:
        st.info("Please upload a dataset to proceed with EDA.")

# Preprocessing Section
elif options == "Preprocessing":
    st.header("Preprocessing")

    if st.session_state["original_data"] is not None:
        original_df = st.session_state["original_data"]
        encoded_df = st.session_state.get("encoded_data", None)
        missing_values_handled_df = st.session_state.get("missing_values_handled_data", None)

        # Encoding Section
        st.subheader("Encode Categorical Features")
        categorical_cols = original_df.select_dtypes(include=["object"]).columns

        if len(categorical_cols) > 0:
            encoding_strategy = st.selectbox(
                "Select encoding strategy", ["OneHot", "Label", "Drop"]
            )
            if st.button("Perform Encoding"):
                encoded_df = encode_categorical(original_df.copy(), encoding_strategy=encoding_strategy)
                st.session_state["encoded_data"] = encoded_df
                st.success("Categorical features encoded!")
                st.write("### Encoded Dataset Preview")
                st.write(encoded_df.head())
        else:
            st.info("No categorical columns found to encode")

        # Handling Missing Values Section
        data_for_missing_values = encoded_df.copy() if encoded_df is not None else original_df.copy()
        missing_values = data_for_missing_values.isnull().sum()
        if missing_values.sum() > 0:
            st.write("### Missing Values")
            st.write(missing_values[missing_values > 0])
            missing_value_strategy = st.selectbox(
                "Select strategy for handling missing values",
                ["mean", "median", "most_frequent"]
            )
            if st.button("Handle Missing Values"):
                if encoded_df is None and len(categorical_cols) > 0:
                    st.warning("Please encode the data to proceed.")
                else:
                    missing_values_handled_df = handle_missing_values(data_for_missing_values.copy(), strategy=missing_value_strategy)
                    st.session_state["missing_values_handled_data"] = missing_values_handled_df
                    st.success("Missing values handled!")
                    st.write("### Updated Dataset (Missing Values Handled)")
                    st.write(missing_values_handled_df.head())
        else:
            st.info("No missing values found.")

        # Feature Selection
        st.subheader("Feature Selection")
        preprocessed_df = missing_values_handled_df if missing_values_handled_df is not None else encoded_df if encoded_df is not None else original_df
        if preprocessed_df is not None:
            features = st.multiselect(
                "Select feature columns (X):",
                options=preprocessed_df.columns,
                default=preprocessed_df.columns[:-1]
            )
            target = st.selectbox(
                "Select target column (y):",
                options=preprocessed_df.columns,
                index=len(preprocessed_df.columns) - 1
            )
            if features and target:
                X = preprocessed_df[features]
                y = preprocessed_df[target]
                st.session_state["features"] = features
                st.session_state["target"] = target

        # Train-Test Split
        st.subheader("Train-Test Split")
        test_size = st.slider(
            "Select test size ratio (0.1 to 0.5):",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05
        )
        random_state = st.number_input(
            "Set random state for reproducibility (optional):",
            value=42,
            step=1
        )
        if st.button("Split Data"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
            st.success(f"Data split completed! Train size: {1 - test_size}, Test size: {test_size}")

        # Scaling Features
        st.subheader("Features Scaling")
        if "X_train" in st.session_state and "X_test" in st.session_state:
            scaling_strategy = st.selectbox(
                "Select scaling strategy:",
                ["StandardScaler", "MinMaxScaler"]
            )
            if st.button("Scale Features"):
                X_train = st.session_state["X_train"]
                X_test = st.session_state["X_test"]
                X_train, X_test = scale_features(X_train, X_test, scaling_strategy)
                st.session_state["X_train"] = X_train
                st.session_state["X_test"] = X_test
                st.success("Features scaled!")
        else:
            st.warning("Please perform train-test split before scaling features.")

        # Final Preprocessed Dataset Preview
        st.header("Preprocessed Dataset Preview")
        st.write(X.head())
    else:
        st.info("Please upload a dataset to proceed with preprocessing.")

# Model Training Section
elif options == "Training and Evaluation":
    st.header("Model Training")

    if st.session_state.get("X_train") is not None and st.session_state.get("y_train") is not None:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        # Task Type Selection
        st.subheader("Select Task Type")
        task_type = st.radio("Select the task type:", ["Regression", "Classification"])
        st.session_state["task_type"] = task_type

        # Model Selection
        st.subheader("Model Selection")
        if task_type == "Regression":
            model_choice = st.selectbox(
                "Select a regression model:",
                ["Linear Regression"]
            )
        elif task_type == "Classification":
            model_choice = st.selectbox(
                "Select a classification model:",
                [
                    "Logistic Regression",
                    "Decision Tree",
                    "SVM",
                    "Naive Bayes",
                    "KNN",
                    "Random Forest",
                    "Gradient Boosting",
                    "AdaBoost",
                ]
            )

        # Train the Model
        if st.button("Train Model"):
            model = None
            if task_type == "Regression":
                if model_choice == "Linear Regression":
                    model = train_linear_regression(X_train, y_train)
            elif task_type == "Classification":
                if model_choice == "Logistic Regression":
                    model = train_logistic_regression(X_train, y_train)
                elif model_choice == "Decision Tree":
                    model = train_decision_tree(X_train, y_train)
                elif model_choice == "SVM":
                    model = train_svm(X_train, y_train)
                elif model_choice == "Naive Bayes":
                    model = train_naive_bayes(X_train, y_train)
                elif model_choice == "KNN":
                    model = train_knn(X_train, y_train)
                elif model_choice == "Random Forest":
                    model = train_random_forest(X_train, y_train)
                elif model_choice == "Gradient Boosting":
                    model = train_gradient_boosting(X_train, y_train)
                elif model_choice == "AdaBoost":
                    model = train_adaboost(X_train, y_train)

            if model:
                st.success(f"{model_choice} trained successfully!")

                # Save the model in session state
                if "trained_models" not in st.session_state:
                    st.session_state["trained_models"] = {}
                st.session_state["trained_models"][model_choice] = model

                # Evaluate the Model
                if task_type == "Regression":
                    st.subheader("Evaluation Metrics")
                    metrics = evaluate_regression(model, X_test, y_test)
                    st.json(metrics)

                elif task_type == "Classification":
                    st.subheader("Evaluation Metrics")
                    metrics, visualizations = evaluate_classification(model, X_test, y_test)
                    st.json(metrics)

                    # Display all visualizations
                    for viz_type, viz_data in visualizations:
                        if viz_type == "Confusion Matrix":
                            st.subheader("Confusion Matrix")
                            cm_fig = plot_confusion_matrix(viz_data)
                            st.pyplot(cm_fig)

                        elif "ROC Curve" in viz_type:
                            st.subheader(viz_type)
                            fpr, tpr, roc_auc = viz_data
                            fig, ax = plt.subplots()
                            ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
                            ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                            ax.set_xlim([0.0, 1.0])
                            ax.set_ylim([0.0, 1.05])
                            ax.set_xlabel("False Positive Rate")
                            ax.set_ylabel("True Positive Rate")
                            ax.set_title("Receiver Operating Characteristic")
                            ax.legend(loc="lower right")
                            st.pyplot(fig)
    else:
        st.info("Please preprocess the data before training models.")

elif options == "Model Comparison":
    st.header("Model Comparison")

    # Check if there are any trained models in session state
    if "trained_models" in st.session_state:
        trained_models = st.session_state["trained_models"]

        # Display all trained models for comparison
        model_names = list(trained_models.keys())  # Get all model names
        selected_models = st.multiselect("Select models to compare:", model_names)

        if selected_models:
            # Get the test data
            X_test = st.session_state.get("X_test")
            y_test = st.session_state.get("y_test")

            # Initialize dictionaries for metrics and visualizations
            metrics_dict = {}

            for model_name in selected_models:
                model = trained_models[model_name]

                # Determine task type based on session state
                task_type = st.session_state["task_type"]

                # Evaluate the model based on its task type
                if task_type == "Regression":
                    metrics = evaluate_regression(model, X_test, y_test)
                elif task_type == "Classification":
                    metrics, _ = evaluate_classification(model, X_test, y_test)
                
                # Store metrics for comparison
                metrics_dict[model_name] = metrics

            # Display the evaluation metrics for the selected models
            st.write("**Evaluation Metrics Comparison:**")
            for model_name, metrics in metrics_dict.items():
                st.subheader(model_name)
                st.json(metrics)
        else:
            st.info("Please select at least one model for comparison.")
    else:
        st.warning("No trained models found. Please train models first.")


# Model Export Section
elif options == "Model Export":
    st.header("Export Trained Models")

    # Check if there are any trained models in session state
    if "trained_models" in st.session_state and st.session_state["trained_models"]:
        trained_models = st.session_state["trained_models"]
        
        # Display all trained models for export
        st.subheader("Select Models to Download")
        model_names = list(trained_models.keys())  # Get all model names
        selected_models = st.multiselect("Select models to download:", model_names)

        if selected_models:
            # Provide a download link for the selected models
            st.subheader("Download Selected Models")
            for model_name in selected_models:
                model = trained_models[model_name]
                with open(f"{model_name}.pkl", "wb") as f:
                    joblib.dump(model, f)
                with open(f"{model_name}.pkl", "rb") as f:
                    st.download_button(
                        label=f"Download {model_name}",
                        data=f,
                        file_name=f"{model_name}.pkl",
                        mime="application/octet-stream"
                    )
                os.remove(f"{model_name}.pkl")  # Remove the file after download
        else:
            st.info("Please select at least one model to download.")
    else:
        st.warning("No trained models found. Please train models first.")

# Cleanup logic for saved models folder
def clear_saved_models_folder():
    folder_path = "saved_models"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)

# Clear the saved models folder when the script stops
clear_saved_models_folder()