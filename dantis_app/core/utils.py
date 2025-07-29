
from PyQt5.QtWidgets import QMessageBox

def show_error(string: str):
    """
    Display a critical error message dialog using PyQt5.

    Parameters
    ----------
    string : str
        The error message to be displayed in the dialog window.
    """
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(string)
    msg.setWindowTitle("Error")
    msg.exec()

def strip_extension(filename):
    """
    Remove the file extension from a given filename.

    Parameters
    ----------
    filename : str
        The name of the file, including its extension.

    Returns
    -------
    str
        The filename without its extension.
    """
    return filename.rsplit('.', 1)[0]

def verify_info(datasets, x_cols, y_cols, models, metrics, val_opts):
    """
    Verifies the presence and validity of all required training information.
    Displays a user-friendly error message for each missing or invalid input.

    Parameters
    ----------
    datasets : dict
        Dictionary containing datasets to be used for training.
    x_cols : dict
        Dictionary mapping dataset IDs to input column selections.
    y_cols : dict
        Dictionary mapping dataset IDs to target variable selections.
    models : dict
        Dictionary containing model configurations and hyperparameters.
    metrics : list
        List of metric names to be used for model evaluation.
    val_opts : dict
        Dictionary containing validation options.

    Returns
    -------
    bool
        True if all required information is present and valid, False otherwise.
    """
    if not datasets:
        show_error(f"No dataset was provided. Model training cannot proceed.")
        return False

    if not x_cols or all(not any(cols.values()) for cols in x_cols.values()):
        show_error(f"No input (X) columns were selected. Model training cannot proceed.")
        return False

    if not y_cols or all(not y for y in y_cols.values()):
        show_error(f"No target (Y) columns were provided. Model training cannot proceed.")
        return False

    if not models:
        show_error(f"NNo model was provided. Model training cannot proceed.")
        return False

    if not metrics:
        show_error(f"No evaluation metric was provided. Model training cannot proceed.")
        return False

    if not val_opts:
        show_error(f"No validation strategy was selected. Model training cannot proceed.")
        return False
    
    return True