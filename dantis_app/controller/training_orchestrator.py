class TrainingOrchestrator:
    """
    Coordinates the preparation of machine learning training workflows by integrating dataset, 
    model, metric, and validation controllers.

    This class serves as a central orchestrator to gather and format all the necessary training
    information in a structured and accessible format. It provides utility methods to extract
    input/output features from datasets, retrieve selected models, metrics, and validation settings.

    Parameters
    ----------
    datasetController : object
        Controller that provides access to all datasets and their structure. Must implement 
        `get_all_datasets()` method and expose `.x_col` and `.y_col` for each dataset.

    modelsController : object
        Controller that contains the available or selected machine learning models. Must have 
        a `models` attribute (typically a list or dict of models).

    metricsController : object
        Controller that holds the selected evaluation metrics. Must expose `checkboxes` attribute 
        representing user-selected metrics.

    validationController : object
        Controller responsible for managing the validation configuration. Must provide a 
        `get_validation_config()` method that returns the validation setup.

    Attributes
    ----------
    datasetController : object
        Instance responsible for accessing and managing dataset configurations and contents.

    modelsController : object
        Instance responsible for providing the list or dictionary of selected model definitions.

    metricsController : object
        Instance containing evaluation metric selections made by the user.

    validationController : object
        Instance that exposes the training validation strategy configuration.

    Methods
    -------
    collect_training_info()
        Collects all necessary information to start training, including datasets, 
        column mappings, selected models, metrics, and validation configuration.

    extract_columns(datasets)
        Extracts the input (X) and output (Y) columns from the datasets, 
        supporting both split datasets and single dataset formats.
    """
    def __init__(self, datasetController, modelsController, metricsController, validationController):
        """
        Initialize the TrainingOrchestrator with the required controller instances.

        Parameters
        ----------
        datasetController : object
            Dataset controller providing access to available datasets.

        modelsController : object
            Controller that holds model definitions and selections.

        metricsController : object
            Controller that manages selected performance metrics.

        validationController : object
            Controller for retrieving validation configuration.
        """
        self.datasetController = datasetController
        self.modelsController = modelsController
        self.metricsController = metricsController
        self.validationController = validationController

    def collect_training_info(self):
        """
        Collect all information needed to prepare and start model training.

        This method retrieves:
        - All datasets (with optional train/val/test splits)
        - X and Y column selections for each dataset
        - Selected models
        - Selected evaluation metrics
        - Validation configuration (e.g., cross-validation folds, type)

        Returns
        -------
        tuple
            A 6-element tuple containing:
            
            datasets : dict
                Dictionary of datasets indexed by unique identifiers.

            x_cols : dict
                Dictionary mapping each dataset to its selected input (X) columns.

            y_cols : dict
                Dictionary mapping each dataset to its output (Y) column(s).

            models : list or dict
                List or dictionary of selected model classes or instances.

            metrics : list
                List of selected metric identifiers or functions.

            validation_options : dict
                Configuration options for validation (e.g., CV folds, stratification).
        """
        datasets = self.datasetController.get_all_datasets()
        validation_options = self.validationController.get_validation_config()
        models = self.modelsController.models
        metrics = self.metricsController.checkboxes
    
        x_cols, y_cols  = self.extract_columns(datasets)
        
        return datasets, x_cols, y_cols, models, metrics, validation_options
    
    def extract_columns(self, datasets):
        """
        Extracts the input and output column mappings from each dataset.

        This method handles both single dataset objects and dataset dictionaries 
        containing train/val/test splits.

        Parameters
        ----------
        datasets : dict
            Dictionary of datasets. Each dataset can either be:
            - A single dataset object with `x_col` and `y_col` attributes
            - A dict with keys like 'train', 'val', 'test', where `train` must have 
              the `x_col` and `y_col` attributes.

        Returns
        -------
        tuple
            A 2-element tuple:
            
            x_cols : dict
                Mapping of dataset ID to its input columns (only those selected by user).

            y_cols : dict
                Mapping of dataset ID to its target column(s).
        """
        x_cols = {}
        y_cols = {}

        for idx, dataset in datasets.items():
            if isinstance(dataset, dict): 
                train_data = dataset['train']
                x_cols[idx] = {k: True for k, v in train_data.x_col.items() if v}
                y_cols[idx] = train_data.y_col
            else: 
                x_cols[idx] = {k: v for k, v in dataset.x_col.items() if v}
                y_cols[idx] = dataset.y_col

        return x_cols, y_cols
    