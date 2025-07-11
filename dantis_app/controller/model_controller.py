from controller.model_discovery import extract_model_names_by_type, extract_default_hyperparameters, instantiate_model_by_name
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from controller.validation_controller import holdout_split, expanding_split, sliding_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils.multiclass import type_of_target
import time, random, os


# --------------------------------------------
# TODO: Temporal fix to allow imports from parent directory 
# This should be removed when the package is public.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# --------------------------------------------
import dantis

class ModelController:
    def __init__(self):
        self.available_models = extract_model_names_by_type(dantis)
        self.params_by_model = extract_default_hyperparameters(dantis)
        self.models = {}
        self.save_path = ""

    def set_path(self, path): 
        self.save_path = path

    def get_path(self):
        return self.save_path

    def get_available_models(self):
        """
        Returns a list of available model types and their default hyperparameters.
        """
        return self.available_models, self.params_by_model

    def add_model(self, model_id, model_instance):
        self.models[model_id] = model_instance

    def delete_model(self, model_id):
        del (self.models[model_id])

    def update_model(self, model_id, model_instance):
        self.models[model_id] = model_instance

    def run_pipeline(self, model_id, X_train, y_train, X_test=None):
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found.")
        
        model = self.models[model_id]
        model.fit(X_train, y_train)
        return model.predict(X_test) if X_test is not None else None


    def instantiate_model(self, model_name, config=None, x=None, y=None, x_test=None, y_test=None):
        """
        Instantiates a model by its name with the given configuration and data.
        """
        return instantiate_model_by_name(model_name, config, x, y, x_test, y_test)


    def split_dataset (self, dataset, x_cols, y_cols, validation_options):
        def extract_xy(ds):
            ## --- 1. Obtener datos base
            X = ds.data[list(x_cols.keys())].to_numpy()
            y = ds.data[y_cols].to_numpy() if y_cols else None
            return X, y

        # Verificamos si dataset es dict (es decir, ya viene separado)
        if isinstance(dataset, dict):
            # Asumimos que contiene las claves: 'train', 'val', 'test'
            X_train, y_train = extract_xy(dataset['train'])
            X_val, y_val = extract_xy(dataset['val'])
            X_test, y_test = extract_xy(dataset['test'])

            # Empaquetamos los folds como una sola tupla train-val
            folds = [((X_train, y_train), (X_val, y_val))]

        else:
            # Caso tradicional: un solo dataset que hay que dividir
            X = dataset.data[list(x_cols.keys())].to_numpy()
            y = dataset.data[y_cols].to_numpy() if y_cols else None

            # --- 2. Validación
            if validation_options['type'] == "train/test":
                train = validation_options["train"] / 100
                validation = validation_options["validation"] / 100
                test = validation_options["test"] / 100
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = holdout_split(X, y, validation, test)
                folds = [((X_train, y_train), (X_val, y_val))]

            elif validation_options['type'] == "crossVal":
                ## REVISAR, LOS FOLDS QUE LE LLEGAN SERAN VACIOS
                k_fold = validation_options["crossVal"]
                percentage_test = validation_options["percentage_crossVal"] / 100
                folds, (X_test, y_test) = expanding_split(X, y, k_fold, percentage_test)

            elif validation_options['type'] == "Sliding_split":
                ## REVISAR, LOS FOLDS QUE LE LLEGAN SERAN VACIOS
                percentage_window = validation_options["sliding"] / 100
                step_size = validation_options["step_size"]
                percentage_sliding = validation_options["percentage_sliding"] / 100
                folds, (X_test, y_test) = sliding_split(X, y, step_size, percentage_window, percentage_sliding)

            else:
                raise ValueError(f"Unknown validation type: {validation_options['type']}")

        return folds, (X_test, y_test)    
    
    def generate_results_by_dataset(self, id_dataset, dataset, y_col, model_name, hyperparameters,
                                    model_path, folds, X_test, y_test, metrics, threshold): 
        """ 
        Generates results based on the provided datasets, models, checkboxes, and validation options. 
        """ 
        results = {} 
        
        # --- 9. Entrenamiento y evaluación
        fold_results = []

        for i, ((X_train, y_train), (X_val, y_val)) in enumerate(folds):
            # model_instance = self.instantiate_model(model_name, config=hyperparameters)
            # model_instance.fit(X_train, y_train)
            if model_path:
                try:
                    if os.path.isfile(model_path) and model_path.endswith(".joblib"):
                        model_instance = self.instantiate_model(model_name, config=hyperparameters)
                        model_instance.load_model(model_path)
                        print(f"Modelo cargado correctamente desde: {model_path}")
                    else:
                        raise FileNotFoundError(f"Archivo no encontrado o extensión inválida: {model_path}")
                except Exception as e:
                    print(f"[WARNING] Fallo al cargar modelo desde {model_path}: {str(e)}")
                    print("[INFO] Entrenando modelo desde cero...")
                    model_instance = self.instantiate_model(model_name, config=hyperparameters)
                    model_instance.fit(X_train, y_train)
            else:
                model_instance = self.instantiate_model(model_name, config=hyperparameters)
                model_instance.fit(X_train, y_train)


            scores = model_instance.predict(X_val)
            predicted_labels = (scores >= threshold).astype(int)

            if y_col[id_dataset] is not None and y_val is not None:
                metric_results = self.evaluate_metrics(y_val, predicted_labels, metrics)
            else:
                metric_results = {"info": "Unsupervised mode - no true labels"}

            fold_results.append({
                "fold": i,
                "scores": scores.tolist(),
                "predictions": predicted_labels.tolist(),
                "metrics": metric_results
            })

        # --- 10. Test final
        test_scores = model_instance.predict(X_test)
        test_predictions = (test_scores >= threshold).astype(int)

        if y_col[id_dataset] is not None and y_val is not None:
            test_metrics = self.evaluate_metrics(y_test, test_predictions, metrics)
        else:
            test_metrics = {"info": "Unsupervised mode - no true labels"}


        # --- 11. Resultado final
        results = {
            "dataset_id": id_dataset,
            "model": model_name,
            "folds": fold_results,
            "test": {
                "scores": test_scores.tolist(),
                "predictions": test_predictions.tolist(),
                "metrics": test_metrics
            }
        }

        os.makedirs("saved_models", exist_ok=True)
        seed = random.randint(0, 99999)
        timestamp = int(time.time())
        seed_time = f"{seed}_{timestamp}"
        model_filename = f"{model_name}_{id_dataset}_{seed_time}.joblib"
        
        path = self.get_path()
        root = os.path.join(path, "DANTIS_Results")
        os.makedirs(root, exist_ok=True)
        save_path = os.path.join(root, model_filename)
        model_instance.save_model(save_path)
        results["model_path"] = save_path

        return results


    def evaluate_metrics(self, y_true, y_pred, metric_list):
        results = {}

        # Verificamos si hay etiquetas verdaderas
        has_ground_truth = len(y_true) != 0 and len(np.unique(y_true)) >= 1

        if not has_ground_truth:
            return {
                metric: "Unavailable (requires ground truth)"
                for metric in metric_list
            }

        # Detectar número de clases
        unique_classes = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
        n_classes = len(unique_classes)
        avg_method = "binary" if n_classes == 2 else "macro"

        # Diccionario de métricas supervisadas
        metric_functions = {
            "Accuracy": lambda yt, yp: accuracy_score(yt, yp),
            "Precision": lambda yt, yp: precision_score(yt, yp, average=avg_method),
            "Recall": lambda yt, yp: recall_score(yt, yp, average=avg_method),
            "F1 Score": lambda yt, yp: f1_score(yt, yp, average=avg_method),
            "True positive": lambda yt, yp: ((yt == 1) & (yp == 1)).sum(),
            "False positive": lambda yt, yp: ((yt == 0) & (yp == 1)).sum(),
            "True negative": lambda yt, yp: ((yt == 0) & (yp == 0)).sum(),
            "False negative": lambda yt, yp: ((yt == 1) & (yp == 0)).sum(),
            "Log-Loss": lambda yt, yp: log_loss(yt, yp),
            "Curva ROC / AUC": lambda yt, yp: roc_auc_score(yt, yp),
        }

        for metric in metric_list:
            func = metric_functions.get(metric)
            if func is not None:
                try:
                    results[metric] = func(y_true, y_pred)
                except Exception as e:
                    results[metric] = f"Error: {str(e)}"
            else:
                results[metric] = "Unsupported metric or not implemented"

        return results

class EvaluationController:
    def __init__(self):
        pass  # Aquí iría el pipeline de entrenamiento, predicción y evaluación

    def run_pipeline(self, model, X_train, y_train, X_test=None):
        model.fit(X_train, y_train)
        return model.predict(X_test) if X_test is not None else None
    
