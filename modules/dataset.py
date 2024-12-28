import os
import pandas as pd
import streamlit as st

# Function to save the uploaded file to the 'data' folder, only if it doesn't already exist
def save_uploaded_file(uploaded_file):
    """
    Save the uploaded dataset to the 'data' folder, if it doesn't already exist.
    
    Args:
        uploaded_file: The uploaded file object.
    
    Returns:
        The path to the saved file.
    """
    # Ensure the 'data' folder exists
    data_folder = 'data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Get the file name and create the full path
    file_path = os.path.join(data_folder, uploaded_file.name)

    # Check if the file already exists
    if os.path.exists(file_path):
        st.write(f"The file '{uploaded_file.name}' exists in the data folder.")
    else:
        # Save the file if it doesn't exist
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"File saved to: {file_path}")
    
    return file_path