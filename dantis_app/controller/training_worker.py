from collections import defaultdict
from PyQt5.QtCore import QObject, pyqtSignal
import pandas as pd
import traceback
from core.utils import strip_extension
from controller.results_formatter import format_results

class TrainingWorker(QObject):
    """
    Worker class that performs model training, evaluation, and results formatting
    for multiple datasets and machine learning models. This class is intended to be
    used in a background thread and communicates results via PyQt signals.

    Signals
    -------
    generate_result : pyqtSignal(str)
        Emitted with formatted result text after processing each dataset.
    notify_training : pyqtSignal()
        Emitted before starting training on each dataset.
    finished : pyqtSignal()
        Emitted when all training tasks have been completed.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets keyed by ID, containing 'train', 'test', and optionally 'val' data.
    x_cols : dict
        Dictionary mapping dataset IDs to the list of input (X) column names.
    y_cols : dict
        Dictionary mapping dataset IDs to the list of target (Y) column names.
    models : dict
        Dictionary of models with associated hyperparameters and metadata.
    metrics : list of str
        List of metric names to evaluate each model.
    options_val : dict
        Validation options used during model training.
    threshold : float
        Threshold value used in certain metric computations.
    modelsController : object
        Controller object responsible for handling model logic.
    controller : object
        Application controller responsible for saving final results.

    Attributes
    ----------
    results_generated : dict
        Dictionary storing evaluation results for each metric.
    final_results : dict
        Dictionary holding structured output per model-dataset combination.

    Methods
    -------
    run()
        Starts the full training and evaluation pipeline.
    get_results()
        Returns the generated evaluation results.
    process_dataset(model_id, model_name, hyperparams, model_path=None)
        Processes a specific model on all datasets.
    """
    generate_result = pyqtSignal(str)
    notify_training = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, datasets, x_cols, y_cols, models,
                 metrics, options_val, threshold, modelsController, controller):
        super().__init__()
        self.datasets = datasets
        self.models = models
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.metrics = metrics
        self.options_val = options_val
        self.threshold = threshold
        self.controller = controller
        self.modelController = modelsController
        self.results_generated = defaultdict(list)
        self.final_results = {}

    def _build_dataset_name(self, dataset):
        """
        Builds a unified name for a dataset using its train, val, and test parts.

        Parameters
        ----------
        dataset : dict
            Dictionary containing 'train', 'val', and 'test' keys.

        Returns
        -------
        str
            A concatenated string representing the dataset.
        """
        parts = [strip_extension(dataset['train'].name)]
        val_info = dataset.get('val')
        if val_info is not None and hasattr(val_info, 'data') and val_info.data is not None and not val_info.data.empty:
            parts.append(strip_extension(val_info.name))
        parts.append(strip_extension(dataset['test'].name))
        return "__".join(parts)
    
    def _build_result_object(self, id_str, dataset_id, dataset, model_name, hyperparams, results):
        """
        Constructs and stores a result object in `final_results`.

        Parameters
        ----------
        id_str : str
            Unique string identifying the dataset-model pair.
        dataset_id : str or int
            ID of the dataset.
        dataset : object or dict
            Dataset object or dictionary with dataset splits.
        model_name : str
            Name of the model used for training.
        hyperparams : dict
            Hyperparameters used for training.
        results : dict
            Output from model evaluation or an error report.
        """
        if results.get("error"):
            self.final_results[id_str] = {
                'dataset_id': dataset_id,
                'dataset': dataset,
                'model': model_name,
                'hyperparameters': hyperparams,
                'error': results['error']
            }
        else:
            self.final_results[id_str] = {
                'dataset_id': dataset_id,
                'dataset': dataset,
                'model': model_name,
                'hyperparameters': hyperparams,
                'folds': results['folds'],
                'test': results['test'],
                'model_path': results['model_path']
            }
            self._update_results(model_name, dataset, results)
    
    def _emit_result(self, id, results):
        """
        Emits a formatted result via the `generate_result` signal.

        Parameters
        ----------
        id : int
            Dataset ID.
        results : dict
            The result data to be formatted and emitted.
        """
        formatted_text = format_results(id, results)
        self.generate_result.emit(formatted_text)
    
    def _initialise_results_structure(self):
        """
        Initializes the result structure to store per-metric outputs.
        """
        for metric in self.metrics:
            if metric not in self.results_generated:
                self.results_generated[metric] = defaultdict(list)

    def _convert_to_dataframe(self):
        """
        Converts accumulated result data into pandas DataFrames per metric.
        """
        for metric in self.metrics:
            if self.results_generated[metric]:
                df = pd.DataFrame(self.results_generated[metric])
                self.results_generated[metric] = df.set_index("Datasets")

    def _update_results(self, model_name, dataset, results):
        """
        Updates the results structure with new metric values from test results.

        Parameters
        ----------
        model_name : str
            Name of the trained model.
        dataset : object or dict
            Dataset used in training.
        results : dict
            Test results containing metric scores.
        """
        name = ""
        if isinstance(dataset, dict) and 'train' in dataset and 'test' in dataset:
            name = self._build_dataset_name(dataset)
        else: 
            name = dataset.name
        for metric in self.metrics:
            valor = results["test"]["metrics"].get(metric)
            self.results_generated[metric][model_name].append(valor if valor is not None else None)

            if name not in self.results_generated[metric]["Datasets"]:
                self.results_generated[metric]["Datasets"].append(name)

    def get_results(self):
        """
        Retrieves the generated results stored internally.

        Returns
        -------
        dict
            Dictionary containing pandas DataFrames of metrics per model.
        """
        return self.results_generated

    def process_dataset(self, model_id, model_name, hyperparams, model_path=None):
        """
        Processes each dataset for a given model, generating results.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model.
        model_name : str
            Name of the model.
        hyperparams : dict
            Dictionary of hyperparameters to apply.
        model_path : str, optional
            Path to a saved model, if applicable.
        """
        if not self.datasets:
            return

        for dataset_id, dataset in self.datasets.items():
            id_str = f"{dataset_id}_{model_id}"

            try:
                self.notify_training.emit()

                folds, (X_test, y_test) = self.modelController.split_dataset(
                    dataset, self.x_cols[dataset_id],
                    self.y_cols[dataset_id], self.options_val
                )

                results = self.modelController.generate_results_by_dataset(
                    dataset_id, dataset, self.y_cols, model_name, hyperparams,
                    model_path, folds, X_test, y_test, self.metrics, self.threshold
                )

                self._build_result_object(
                    id_str, dataset_id, dataset, model_name, hyperparams, results
                )   

            except Exception as e:
                error_msg = f"Error procesando dataset '{dataset_id}' con modelo '{model_id}':\n{traceback.format_exc()}"
                self.final_results[id_str] = {
                    'dataset_id': dataset_id,
                    'dataset': dataset,
                    'model': model_name,
                    'hyperparameters': hyperparams,
                    'error': error_msg
                }
                if hasattr(self, 'error'):
                    self.error.emit(error_msg)

            finally:
                self._emit_result(dataset_id, self.final_results[id_str])

    def run(self):
        self._initialise_results_structure()

        for model_id, model_data in self.models.items():
                model_name = model_data['model']
                hyperparams = model_data['hyperparameters']
                model_path = model_data.get('paths', None)

                self.process_dataset(model_id, model_name, hyperparams, model_path)

        self.controller.set_save_results(self.final_results)
        self._convert_to_dataframe()
        self.finished.emit()