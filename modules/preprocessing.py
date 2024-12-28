import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import streamlit as st

def handle_missing_values(df, strategy="mean"):
    """
    Handle missing values by imputing with mean, median, or most frequent.
    """
    if df.isnull().sum().sum() == 0:
        st.warning("No missing values present in the dataset.")
        return df
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

def encode_categorical(df, encoding_strategy="OneHot"):
    """
    Encode categorical columns using OneHotEncoder or LabelEncoder.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    if categorical_cols.empty:
        st.warning("No categorical columns found in the dataset.")
        return df

    if encoding_strategy == "OneHot":
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated argument
        encoded_df = pd.DataFrame(
            encoder.fit_transform(df[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols)
        )
        df = df.drop(categorical_cols, axis=1)
        return pd.concat([df, encoded_df], axis=1)
    elif encoding_strategy == "Label":
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
        return df
    elif encoding_strategy == "Drop":
        return df.drop(columns=categorical_cols)

def scale_features(X_train, X_test, scaling_strategy):
    """
    Scales the feature columns in the training and testing datasets.

    Parameters:
        X_train (DataFrame): Training feature set.
        X_test (DataFrame): Testing feature set.
        scaling_strategy (str): Scaling strategy to use. Options are "StandardScaler" or "MinMaxScaler".

    Returns:
        X_train_scaled, X_test_scaled: Scaled training and testing feature sets.
    """
    if scaling_strategy == "StandardScaler":
        scaler = StandardScaler()
    elif scaling_strategy == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaling strategy. Choose 'StandardScaler' or 'MinMaxScaler'.")

    # Fit the scaler on the training data and transform both train and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled
