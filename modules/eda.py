import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def plot_data_distribution(df):
    """
    Plot data distribution for all numerical columns.
    """
    st.write("### Data Distribution")
    numerical_cols = df.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 0:
        for col in numerical_cols:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df[col], kde=True, bins=30, ax=ax)
            ax.set_title(f"Data Distribution of {col}")
            st.pyplot(fig)
    else:
        st.warning("No numerical columns found for data distribution.")

def plot_histogram(df, column):
    """
    Plot histogram for the selected column.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df[column], kde=True, bins=30, ax=ax)
    ax.set_title(f"Histogram of {column}")
    st.pyplot(fig)

def plot_boxplot(df, column):
    """
    Plot boxplot for the selected column.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f"Boxplot of {column}")
    st.pyplot(fig)

def plot_corr_heatmap(df):
    """
    Plot a correlation heatmap for numerical columns.
    """
    corr = df.select_dtypes(include=['number']).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

def plot_pairplot(df):
    """
    Plot pairplot for numerical columns.
    """
    numerical_cols = df.select_dtypes(include=['number'])
    if len(numerical_cols.columns) > 1:
        # Pairplot creates its own figure, so we don't use plt.subplots here
        fig = sns.pairplot(numerical_cols).fig
        st.pyplot(fig)
    else:
        st.warning("Pairplot requires at least two numerical columns.")
