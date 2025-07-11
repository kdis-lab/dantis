

## ESTE ARCHIVO SE PUEDE BORRAR, CREO QUE NO SE USA EN NINGUN SITIO
class DatasetManager:
    def __init__(self, datasetController, validationController):
        self.datasetController = datasetController
        self.validationController = validationController

    def build_datasets(self):
        original_datasets = self.datasetController.datos
        split_config = self.validationController.get_info_tabla()
        datasets = self._associate_split_datasets(original_datasets, split_config)
        return datasets

    def _associate_split_datasets(self, datasets, split_config):
        final_split_datasets = {}

        if split_config:
            for group_id, splits in split_config.items():
                final_split_datasets[group_id] = {}
                for split_name, file_name in splits.items():
                    for key, ds_info in list(datasets.items()):
                        if ds_info.name == file_name:
                            final_split_datasets[group_id][split_name] = ds_info
                            del datasets[key]
                            break
                        
        return datasets, final_split_datasets
    
    def extract_columns(self, datasets):
        x_cols = {info.id: {k: v for k, v in info.x_col.items() if v} for info in datasets.values()}
        y_cols = {info.id: info.y_col for info in datasets.values()}
        return x_cols, y_cols
    
    def extract_columns_split(self, split_datasets):
        x_cols = {
            dataset_id: {k: True for k, v in d['train'].x_col.items() if v}
            for dataset_id, d in split_datasets.items()
        }

        y_cols = {dataset_id: d['train'].y_col for dataset_id, d in split_datasets.items()}
        return x_cols, y_cols