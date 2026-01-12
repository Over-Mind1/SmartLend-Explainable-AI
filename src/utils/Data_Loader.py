from src.config.config import TRAIN_DATA_PATH, X_TEST_DATA_PATH, Y_TEST_DATA_PATH
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def data_loader(
    val_size: float = 0.1,
    test_size: float = 0.2,
    target_name: str = "Default",
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series
]:
    """
    Load dataset and return train / validation / test splits.

    Args:
        val_size (float): validation size from training data
        target_name (str): target column name
        random_state (int): random seed

    Returns:
        x_train, x_val, x_test, y_train, y_val, y_test
    """
    # Load training data
    train_data = pd.read_csv(TRAIN_DATA_PATH)

    # Load test data
    X_test = pd.read_csv(X_TEST_DATA_PATH)
    y_test = pd.read_csv(Y_TEST_DATA_PATH).rename(
        columns={"predicted_probability": target_name}
    )

    # Concatenate X_test and y_test based on LoanID
    full_test_data = pd.merge(
        X_test,
        y_test,
        on="LoanID",
        how="inner"  
    )

    # Combine train data and test data
    full_data = pd.concat(
        [train_data, full_test_data],
        axis=0,
        ignore_index=True
    ).drop(columns=["LoanID"])
    # Separate features and target
    X_full = full_data.drop(columns=[target_name])
    y_full = full_data[target_name]


    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=random_state, stratify=y_full
    )
    # Then split the remaining data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )   

    return x_train, x_val, X_test, y_train, y_val, y_test
