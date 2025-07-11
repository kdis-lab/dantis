class TrainingOrchestrator:
    def __init__(self, datasetController, modelsController, metricsController, validationController):
        self.datasetController = datasetController
        self.modelsController = modelsController
        self.metricsController = metricsController
        self.validationController = validationController

    def collect_training_info(self):

        datasets = self.datasetController.get_all_datasets()
        validation_options = self.validationController.get_validation_config()
        models = self.modelsController.models
        metrics = self.metricsController.checkboxes
    
        # For complete datasets
        x_cols, y_cols  = self.extract_columns(datasets)
        
        return datasets, x_cols, y_cols, models, metrics, validation_options
    
    def extract_columns(self, datasets):
        x_cols = {}
        y_cols = {}

        for idx, dataset in datasets.items():
            if isinstance(dataset, dict):  # Caso con splits: train/val/test
                train_data = dataset['train']
                x_cols[idx] = {k: True for k, v in train_data.x_col.items() if v}
                y_cols[idx] = train_data.y_col
            else:  # DatasetInfo único
                x_cols[idx] = {k: v for k, v in dataset.x_col.items() if v}
                y_cols[idx] = dataset.y_col

        return x_cols, y_cols
    