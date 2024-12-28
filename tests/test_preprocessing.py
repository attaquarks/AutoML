import pytest
import pandas as pd
from ..modules.preprocessing import encode_categorical

def test_encode_categorical():
    df = pd.DataFrame({
        'color': ['red', 'blue', 'green'],
        'size': ['small', 'large', 'medium']
    })
    encoded_df = encode_categorical(df, 'OneHot')
    assert encoded_df.shape[1] == 5  # Should have 5 columns after one-hot encoding

def test_handle_missing_values():
    df = pd.DataFrame({
        'age': [25, None, 30, 35],
        'salary': [50000, 60000, None, 70000]
    })
    df_filled = handle_missing_values(df, 'mean')
    assert df_filled['age'].isnull().sum() == 0  # Should have no missing values in 'age'
    assert df_filled['salary'].isnull().sum() == 0  # Should have no missing values in 'salary'
