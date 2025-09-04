import time
import random
import numpy as np
from controller.model_discovery import (
    extract_model_names_by_type,
    extract_default_hyperparameters,
    instantiate_model_by_name,
)
from controller.validation_controller import (
    holdout_split,
    expanding_split,
    sliding_split,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

# --------------------------------------------
# TODO: Temporal fix to allow imports from parent directory 
# This should be removed when the package is public.
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import dantis
# --------------------------------------------

class ModelController:
    """
    Controller class to manage model instantiation, training, validation, and evaluation.

    This class wraps around DANTIS model discovery and validation mechanisms, providing
    a complete interface to manage datasets, models, metrics, and results.

    Attributes
    ----------
    available_models : list
        List of model names available in the DANTIS package.
    params_by_model : dict
        Dictionary of default hyperparameters for each available model.
    models : dict
        Dictionary mapping model IDs to instantiated models.
    save_path : str
        Path to save trained model files.

    Methods
    -------
    set_path(path)
        Set the save path for model outputs.
    
    get_path()
        Return the current model save path.

    get_models()
        Retrieve the dictionary of added models.

    get_available_models()
        Return available model names and their hyperparameters.

    add_model(model_id, model_instance)
        Add a new model to the controller.

    delete_model(model_id)
        Remove a model from the controller.

    update_model(model_id, model_instance)
        Replace an existing model with a new one.

    run_pipeline(model_id, X_train, y_train, X_test=None)
        Train a model and return predictions if test data is provided.

    instantiate_model(model_name, config=None, x=None, y=None, x_test=None, y_test=None)
        Create a model using its name and an optional configuration.

    split_dataset(dataset, x_cols, y_cols, validation_options)
        Split dataset based on selected validation strategy (holdout, cross-validation, etc.).

    generate_results_by_dataset(...)
        Train and evaluate models using provided datasets and return structured results.

    evaluate_metrics(y_true, y_pred, metric_list)
        Evaluate selected metrics from predictions and true labels.
    """
    def __init__(self):
        """
        Initialize a new ModelController instance.

        Loads available models and their hyperparameters from DANTIS.
        """
        self.available_models = extract_model_names_by_type(dantis)
        self.params_by_model = extract_default_hyperparameters(dantis)
        self.models = {}
        self.save_path = ""

    def set_path(self, path): 
        """
        Set the save path for trained models.

        Parameters
        ----------
        path : str
            Path to the directory where models will be saved.
        """
        self.save_path = path

    def get_path(self):
        """
        Get the current model save path.

        Returns
        -------
        str
            Path where models will be saved.
        """
        return self.save_path
    
    def get_models(self): 
        """
        Retrieve the stored models.

        Returns
        -------
        dict
            Dictionary of model instances.
        """
        return self.models

    def get_available_models(self):
        """
        Get available models and their default hyperparameters.

        Returns
        -------
        tuple
            A tuple (model_names, default_params).
        """
        return self.available_models, self.params_by_model

    def add_model(self, model_id, model_instance):
        """
        Add a model to the controller.

        Parameters
        ----------
        model_id : str
            Identifier for the model.
        model_instance : object
            Instantiated model.
        """
        self.models[model_id] = model_instance

    def delete_model(self, model_id):
        """
        Remove a model from the controller.

        Parameters
        ----------
        model_id : str
            ID of the model to remove.
        """
        del (self.models[model_id])

    def update_model(self, model_id, model_instance):
        """
        Update or replace an existing model.

        Parameters
        ----------
        model_id : str
            Identifier for the model.
        model_instance : object
            New model instance to update.
        """
        self.models[model_id] = model_instance

    def run_pipeline(self, model_id, X_train, y_train, X_test=None):
        """
        Train and predict using a specified model.

        Parameters
        ----------
        model_id : str
            Identifier for the model to use.
        X_train : array-like
            Training data features.
        y_train : array-like
            Training data labels.
        X_test : array-like, optional
            Test data to predict on.

        Returns
        -------
        array-like or None
            Predictions on test data, or None if X_test is not provided.
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found.")
        
        model = self.models[model_id]
        model.fit(X_train, y_train)
        return model.predict(X_test) if X_test is not None else None

    def instantiate_model(self, model_name, config=None, x=None, y=None, x_test=None, y_test=None):
        """
        Instantiate a model using its name and optional parameters.

        Parameters
        ----------
        model_name : str
            Name of the model class.
        config : dict, optional
            Hyperparameter configuration.
        x, y, x_test, y_test : array-like, optional
            Optional datasets to pass to the model constructor.

        Returns
        -------
        object
            Instantiated model.
        """
        return instantiate_model_by_name(model_name, config, x, y, x_test, y_test)

    def split_dataset (self, dataset, x_cols, y_cols, validation_options):
        """
        Split the dataset according to the selected validation strategy.

        Parameters
        ----------
        dataset : object or dict
            Dataset object or pre-split dictionary.
        x_cols : dict
            Mapping of input feature names.
        y_cols : str
            Name of the output column.
        validation_options : dict
            Validation configuration including type and split percentages.

        Returns
        -------
        tuple
            (folds, (X_test, y_test)) for training and testing.
        """
        def extract_xy(ds):
            X = ds.data[list(x_cols.keys())].to_numpy(dtype=float)
            y = ds.data[y_cols].to_numpy(dtype=float) if y_cols else None
            return X, y

        if isinstance(dataset, dict):
            X_train, y_train = extract_xy(dataset['train'])
            X_val, y_val = extract_xy(dataset.get('val')) if dataset.get('val') else (None, None)
            X_test, y_test = extract_xy(dataset['test'])
            folds = [((X_train, y_train), (X_val, y_val))]
        else:
            X, y = extract_xy(dataset)
            val_type = validation_options.get('type')
            if val_type == "train/test":
                validation = validation_options["validation"] / 100
                test = validation_options["test"] / 100
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = holdout_split(X, y, validation, test)
                folds = [((X_train, y_train), (X_val, y_val))]

            elif val_type == "crossVal":
                k_fold = validation_options["crossVal"]
                pct_test = validation_options["percentage_crossVal"] / 100
                folds, (X_test, y_test) = expanding_split(X, y, k_fold, pct_test)

            elif val_type == "Sliding_split":
                pct_window = validation_options["sliding"] / 100
                pct_slide = validation_options["percentage_sliding"] / 100
                folds, (X_test, y_test) = sliding_split(X, y, pct_window, pct_slide)

            else:
                raise ValueError(f"Unknown validation type: {val_type}")

        return folds, (X_test, y_test)    

    def generate_results_by_dataset(self, id_dataset, dataset, y_col, model_name, hyperparameters,
                                    model_path, folds, X_test, y_test, metrics, threshold): 
        """
        Generate results from training, validation, and test predictions.

        Parameters
        ----------
        id_dataset : str
            Dataset identifier.
        dataset : object
            Dataset used for training/validation/test.
        y_col : dict
            Target columns by dataset ID.
        model_name : str
            Name of the model to instantiate.
        hyperparameters : dict
            Model hyperparameters.
        path : str
            Path to pre-trained model, if available.
        folds : list
            List of training and validation fold tuples.
        X_test : array-like
            Features for test evaluation.
        y_test : array-like
            Labels for test evaluation.
        metrics : list
            List of evaluation metrics.
        threshold : float
            Threshold to binarize predicted scores.

        Returns
        -------
        dict
            Structured dictionary of model evaluation results.
        """
        results = {} 
        fold_results = []
        for i, ((X_train, y_train), (X_val, y_val)) in enumerate(folds):
            model_instance = self._get_or_train_model(model_name, hyperparameters, model_path, X_train, y_train)
            if model_instance is None:
                results = {
                    "dataset_id": id_dataset,
                    "error": "Valores insuficientes para entrenamiento"
                }
                return results
            if X_val is not None and y_val is not None and len(X_val) > 0:
                scores = model_instance.predict(X_val)
                predicted_labels = (scores >= threshold).astype(int)
                if y_col[id_dataset] is not None:
                    metric_results = self.evaluate_metrics(y_val, predicted_labels, metrics)
                else:
                    metric_results = {"info": "Unsupervised mode - no true labels"}
                fold_results.append({
                    "fold": i,
                    "scores": scores.tolist(),
                    "predictions": predicted_labels.tolist(),
                    "metrics": metric_results
                })
            else:
                fold_results.append({
                    "fold": i,
                    "info": "Sin validación",
                    "metrics": None
                })
        # --- Test evaluation ---
        if X_test is not None and len(X_test) > 0:
            test_scores = model_instance.predict(X_test)
            test_preds  = (test_scores >= threshold).astype(int)

            test_metrics = (
                self.evaluate_metrics(y_test, test_preds, metrics)
                if y_col[id_dataset] and y_test is not None
                else {"info": "Unsupervised mode - no labels"}
            )
        else:
            test_scores, test_preds = [], []
            test_metrics = {"info": "No se proporcionó conjunto de test o está vacío"}

        # --- Final result ---
        results = {
            "dataset_id": id_dataset,
            "model": model_name,
            "folds": fold_results,
            "test": {
                "scores": test_scores if isinstance(test_scores, list) else test_scores.tolist(),
                "predictions": test_preds if isinstance(test_preds, list) else test_preds.tolist(),
                "metrics": test_metrics
            }
        }
        # --- Save model ---
        results["model_path"] = self._save_model(model_instance, model_name, id_dataset)
        return results

    def _get_or_train_model(self, model_name, hyperparameters, path, X_train, y_train):
        """
        Loads a model from disk if available; otherwise, instantiates and trains a new model.

        This method first checks if a valid path to a saved model is provided. If so, it attempts 
        to load the model. If loading fails or no path is provided, it creates a new instance of 
        the model using the provided name and hyperparameters and trains it on the given data.

        If the training data contains only one class, model training is skipped, and `None` is returned.

        Parameters
        ----------
        model_name : str
            The name of the model to instantiate or load.

        hyperparameters : dict
            Dictionary of hyperparameters to use for model instantiation.

        path : str
            Path to the saved model file. Must be a `.joblib` file.

        X_train : array-like
            Feature matrix for training.

        y_train : array-like
            Target vector for training.

        Returns
        -------
        model : object or None
            Trained model instance, or None if model could not be trained (e.g., only one class in `y_train`).
        """
        if path and os.path.isfile(path) and path.endswith(".joblib"):
            try:
                model = self.instantiate_model(model_name, config=hyperparameters)
                model.load_model(path)
                return model
            except Exception as e:
                print(f"[WARNING] Fallo al cargar modelo desde {path}: {str(e)}")
                print("[INFO] Entrenando modelo desde cero...")

        if len(np.unique(y_train)) == 1:
            return None  
        
        model = self.instantiate_model(model_name, config=hyperparameters)
        model.fit(X_train, y_train)
        return model

    def _save_model(self, model_instance, model_name, dataset_id):
        """
        Saves a trained model to disk using a unique filename.

        The model is saved under the directory specified by the `save_path` attribute, inside a
        subfolder named `DANTIS_Results`. The filename includes the model name, dataset ID, a 
        random integer, and a timestamp to ensure uniqueness.

        Parameters
        ----------
        model_instance : object
            The model instance to be saved. Must implement a `.save_model(path)` method.

        model_name : str
            Name of the model (used in the filename).

        dataset_id : str or int
            Identifier of the dataset associated with this model (used in the filename).

        Returns
        -------
        full_path : str
            The full path where the model was saved.
        """
        os.makedirs("saved_models", exist_ok=True)
        seed_time = f"{random.randint(0, 99999)}_{int(time.time())}"
        filename = f"{model_name}_{dataset_id}_{seed_time}.joblib"
        save_dir = os.path.join(self.save_path, "DANTIS_Results")
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)
        model_instance.save_model(full_path)
        return full_path

    def evaluate_metrics(self, y_true, y_pred, metric_list):
        """
        Compute metrics given true and predicted labels.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.
        y_pred : array-like
            Predicted labels.
        metric_list : list of str
            Names of metrics to compute.

        Returns
        -------
        dict
            Dictionary mapping metric names to computed values.
        """
        if len(y_true) == 0 or len(np.unique(y_true)) < 1:
            return {metric: "Unavailable (requires ground truth)" for metric in metric_list}
        n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
        avg = "binary" if n_classes == 2 else "macro"
        
        metric_funcs = {
            "Accuracy": lambda yt, yp: accuracy_score(yt, yp),
            "Precision": lambda yt, yp: precision_score(yt, yp, average=avg),
            "Recall": lambda yt, yp: recall_score(yt, yp, average=avg, zero_division=0),
            "F1 Score": lambda yt, yp: f1_score(yt, yp, average=avg),
            "True positive": lambda yt, yp: ((yt == 1) & (yp == 1)).sum(),
            "False positive": lambda yt, yp: ((yt == 0) & (yp == 1)).sum(),
            "True negative": lambda yt, yp: ((yt == 0) & (yp == 0)).sum(),
            "False negative": lambda yt, yp: ((yt == 1) & (yp == 0)).sum(),
            "Log-Loss": lambda yt, yp: log_loss(yt, yp),
            "Curva ROC-AUC": lambda yt, yp: roc_auc_score(yt, yp),
        }
        results = {}
        for metric in metric_list:
            func = metric_funcs.get(metric)
            try:
                results[metric] = func(y_true, y_pred) if func else "Unsupported metric"
            except Exception as e:
                results[metric] = f"Error: {e}"
        return results

class EvaluationController:
    """
    Controller class for running basic model training and evaluation.

    Methods
    -------
    run_pipeline(model, X_train, y_train, X_test=None)
        Train a model and optionally return test predictions.
    """
    def __init__(self):
        """
        Initialize a basic EvaluationController.
        """
        pass 

    def run_pipeline(self, model, X_train, y_train, X_test=None):
        """
        Train and return model predictions if test data is provided.

        Parameters
        ----------
        model : object
            Model with fit and predict methods.
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
        X_test : array-like, optional
            Test features.

        Returns
        -------
        array-like or None
            Predictions or None if X_test is not provided.
        """
        model.fit(X_train, y_train)
        return model.predict(X_test) if X_test is not None else None