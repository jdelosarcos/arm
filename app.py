import streamlit as st
import pandas as pd

st.title("Market Basket Analysis Dashboard")
st.write("Analyze customer purchasing patterns and discover product associations")

# Helper Functions

def preprocess_data(df):
    """
    Preprocess data to make it suitable for apriori algorithm.
    Handles different types of input data formats.
    
    # Convert to boolean/binary if not already
    df = df.astype(bool).astype(int)
    Args:
        df (pandas.DataFrame): Input DataFrame
    Returns:
        pandas.DataFrame: Preprocessed DataFrame with binary values
    """
    # Make a copy to avoid modifying original data
    df_copy = df.copy()
    
    # Check if the data is already in the correct format (binary/boolean)
    if set(df_copy.values.ravel()).issubset({0, 1, True, False, 'True', 'False', '0', '1'}):
        # Convert to boolean then to int (0,1)
        for col in df_copy.columns:
            df_copy[col] = df_copy[col].map({'True': True, 'False': False, 
                                           '1': True, '0': False, 
                                           1: True, 0: False}).astype(int)
        return df_copy
    
    # Check if we have numerical data (e.g., quantity-based)
    if df_copy.select_dtypes(include=['int64', 'float64']).columns.any():
        # Convert to binary (1 if quantity > 0, else 0)
        df_copy = (df_copy > 0).astype(int)
        return df_copy
    
    # For categorical/text data, convert to one-hot encoded format
    if df_copy.select_dtypes(include=['object']).columns.any():
        try:
            # First, try to handle comma-separated values
            if df_copy.iloc[0].str.contains(',').any():
                # Split comma-separated values and create one-hot encoding
                all_items = set()
                for col in df_copy.columns:
                    items = df_copy[col].str.split(',').explode().str.strip()
                    all_items.update(items.unique())
                
                # Remove any empty or null values
                all_items = {item for item in all_items if item and pd.notna(item)}
                
                # Create binary columns for each unique item
                binary_df = pd.DataFrame(index=df_copy.index)
                for item in sorted(all_items):
                    binary_df[item] = df_copy.apply(
                        lambda row: any(item in str(val).split(',') for val in row), 
                        axis=1
                    ).astype(int)
                return binary_df
            
            # If not comma-separated, treat each unique value as a separate column
            else:
                binary_df = pd.get_dummies(df_copy).astype(int)
                return binary_df
                
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")
            return None

    return None

def validate_data(df):
    """Validate that data contains only binary values."""
    return ((df == 0) | (df == 1)).all().all()

def load_data(file):
    """
    Load and preprocess the data file.
    
    Args:
        file: Uploaded file object
    Returns:
        pandas.DataFrame: Preprocessed DataFrame suitable for apriori algorithm
    """
    try:
        # Check file extension
        file_extension = file.name.split('.')[-1].lower()
        
        # Read the file based on its extension
        if file_extension == 'csv':
            # Try different encodings
            try:
                df = pd.read_csv(file)
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding='latin1')
                
            # Remove any unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # If first column looks like an index, use it as index
            if df.columns[0].lower() in ['index', 'id', 'transaction_id', 'transaction']:
                df.set_index(df.columns[0], inplace=True)
            
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Preprocess the data
        processed_df = preprocess_data(df)
        
        if processed_df is None:
            st.error("Could not process the data into the required format.")
            return None
            
        # Validate final format
        if not validate_data(processed_df):
            st.error("Processed data is not in the correct format (binary values only).")
            return None
            
        return processed_df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
