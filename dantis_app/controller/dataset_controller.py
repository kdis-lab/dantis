import requests
import os
import pandas as pd
import io
import arff
from dataclasses import dataclass, field
import uuid
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict
from urllib.parse import urlparse

@dataclass
class DatasetInfo:
    """
    Data container for storing dataset information.

    Attributes
    ----------
    id : int
        Unique identifier for the dataset (UUID).
    name : str
        Name of the dataset.
    path : str
        Source path or URL of the dataset.
    data : pandas.DataFrame
        The dataset loaded as a DataFrame.
    x_col : dict, optional
        Dictionary of feature columns configuration, defaults to empty dict.
    y_col : str, optional
        Name of the target column, defaults to None.
    """
    id: int 
    name: str
    path: str
    data: pd.DataFrame
    x_col: dict = field(default_factory=dict)
    y_col: str = None    

@dataclass
class PartitionGroup:
    """
    Represents a group of dataset partitions (train, validation, test).

    Attributes
    ----------
    id : int
        Unique identifier for the partition group (UUID).
    name : str
        Name of the partition group.
    train_id : Optional[int], optional
        Dataset ID used for training partition, defaults to None.
    val_id : Optional[int], optional
        Dataset ID used for validation partition, defaults to None.
    test_id : Optional[int], optional
        Dataset ID used for testing partition, defaults to None.
    """
    id: int  
    name: str
    train_id: Optional[int] = None
    val_id: Optional[int] = None
    test_id: Optional[int] = None

COMMON_ENCODINGS = ["utf-8", "latin1", "ISO-8859-1", "utf-16"]
COMMON_SEPARATORS = [",", ";", "\t"]

def _try_read_csv(content, from_file=True):
    """
    Attempts to read CSV data trying multiple encodings and separators until successful.

    Parameters
    ----------
    content : str
        File path if from_file is True, or CSV content string if False.
    from_file : bool, optional
        Indicates if content is a file path (True) or CSV text content (False), by default True.

    Returns
    -------
    pandas.DataFrame
        DataFrame parsed from the CSV content.

    Raises
    ------
    ValueError
        If no valid CSV parsing could be performed with common encodings and separators.
    """
    for encoding in COMMON_ENCODINGS:
        for sep in COMMON_SEPARATORS:
            try:
                if from_file:
                    df = pd.read_csv(content, encoding=encoding, sep=sep)
                else:
                    df = pd.read_csv(io.StringIO(content), encoding=encoding, sep=sep)
                if not df.empty:
                    return df
            except Exception:
                continue
    raise ValueError("No se pudo leer el archivo CSV con codificadores y separadores comunes.")

def _load_file(path_or_url: str, is_remote=False) -> pd.DataFrame:
    """
    Load a dataset from local file or remote URL in supported formats.

    Supports CSV, JSON, Excel, Parquet, and ARFF file formats.

    Parameters
    ----------
    path_or_url : str
        Path or URL to the dataset file.
    is_remote : bool, optional
        Whether the source is remote (URL) or local file, by default False.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset as a DataFrame.

    Raises
    ------
    ValueError
        If file format is not supported.
    """
    ext = os.path.splitext(path_or_url)[-1].lower()
    
    if is_remote:
        response = requests.get(path_or_url, timeout=10)
        response.raise_for_status()
        content = response.content if ext != ".csv" else response.text
    else:
        content = path_or_url

    if ext == ".csv":
        return _try_read_csv(content, from_file=not is_remote)
    elif ext == ".json":
        return pd.read_json(path_or_url) if not is_remote else pd.read_json(io.StringIO(response.text))
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(path_or_url) if not is_remote else pd.read_excel(io.BytesIO(response.content))
    elif ext == ".parquet":
        return pd.read_parquet(path_or_url) if not is_remote else pd.read_parquet(io.BytesIO(response.content))
    elif ext == ".arff":
        if is_remote and response:
            text = response.content.decode("utf-8", errors="ignore")
            arff_data = arff.load(io.StringIO(text))
        else:
            with open(path_or_url, 'r', encoding='utf-8', errors='ignore') as f:
                arff_data = arff.load(f)
        attributes = [attr[0] for attr in arff_data['attributes']]
        return pd.DataFrame(arff_data['data'], columns=attributes)
    else:
        raise ValueError(f"Formato no soportado: {ext}")

def load_predefined_datasets_name(github_repo_url):
    """
    Loads the list of available dataset filenames from a GitHub repository URL.

    Parameters
    ----------
    github_repo_url : str
        URL to the GitHub API endpoint returning the repository contents.

    Returns
    -------
    list of str
        List all dataset filenames in the repository.
    """
    try:
        response = requests.get(github_repo_url)

        if response.status_code == 200:
            datasets = response.json() 
            return [file.get("name", "") for file in datasets]
        else:
            print(f"Failed to get datasets: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"Error processing request: {e}")
        return []

class DatasetController:
    """
    Controller class to manage dataset loading, registration, partitioning, and metadata handling.

    Attributes
    ----------
    data : Dict[int, DatasetInfo]
        Dictionary storing datasets indexed by internal integer IDs.
    partitioned_groups : Dict[str, PartitionGroup]
        Dictionary storing dataset partition groups by unique string keys.
    _actual_id : int
        Internal counter to assign unique incremental dataset IDs.
    options_x_col : dict
        Currently selected feature columns options.
    y_col : str or None
        Currently selected target column.

    Methods
    -------
    _register_dataset(name, path, data)
        Registers a dataset and assigns a unique ID.
    _load_data_local_o_predefinido(file_path)
        Loads local or predefined dataset from file path.
    _load_data_url(url)
        Loads dataset from remote URL.
    add_datasets(file_path, tipo)
        Adds a dataset from local/predefined/URL source.
    delete(id)
        Deletes a dataset by its ID.
    _reestructurar_ids()
        Restructures dataset IDs sequentially after deletion.
    get_data()
        Returns dictionary of all datasets.
    get_split_dataset()
        Returns dictionary of partition groups.
    get_all_datasets()
        Returns all datasets, grouping partitioned datasets.
    get_all_datasets_names()
        Returns names of datasets and partition groups.
    set_split_data(data)
        Sets partition groups by mapping dataset names to IDs.
    get_columns(id)
        Returns column names for a dataset by ID.
    set_options_x_col(options_x_col)
        Sets feature columns options.
    get_options_x_col(id)
        Gets feature columns options for a dataset.
    set_y_col(y_col)
        Sets the target column.
    get_y_col(id)
        Gets the target column for a dataset.
    add_x_col(id, options_x_col)
        Assigns feature columns to a dataset.
    add_y_col(id, y_col)
        Assigns target column to a dataset.
    """
    def __init__(self):
        self.data: Dict[int, DatasetInfo] = {}
        self.partitioned_groups: Dict[str, PartitionGroup] = {}
        self._actual_id = 1   
        self.options_x_col = {}
        self.y_col = None

    def _register_dataset(self, name: str, path: str, data: pd.DataFrame):
        """
        Registers a new dataset with a unique internal ID.

        Parameters
        ----------
        name : str
            Name of the dataset.
        path : str
            Source path or URL of the dataset.
        data : pandas.DataFrame
            Dataset content.

        Returns
        -------
        DatasetInfo
            Registered dataset information object.
        """
        if self.data:
            last_id = max(self.data.keys())
            self._actual_id = last_id + 1
        info = DatasetInfo(id=uuid.uuid4(), name=name, path=path, data=data)
        self.data[self._actual_id] = info
        return info
    
    def _load_data_local_or_predefined(self, file_path:str):
        """
        Loads a dataset from a local or predefined file path.

        Parameters
        ----------
        file_path : str
            Path to the local dataset file.

        Returns
        -------
        DatasetInfo
            Registered dataset info.
        """
        is_remote = False
        def is_remote_path(path):
            return urlparse(path).scheme in ('http', 'https')

        is_remote = is_remote_path(file_path)

        data = _load_file(file_path, is_remote)
        return self._register_dataset(name=os.path.basename(file_path), path=file_path, data=data)

    def _load_data_url(self, url: str):
        """
        Loads a dataset from a remote URL.

        Parameters
        ----------
        url : str
            URL to the dataset.

        Returns
        -------
        DatasetInfo
            Registered dataset info.
        """
        data = _load_file(url, is_remote=True)
        return self._register_dataset(name=os.path.basename(url), path=url, data=data)

    def add_datasets(self, file_path, _type):
        """
        Adds a dataset from local, predefined, or URL source based on type indicator.

        Parameters
        ----------
        file_path : str
            Path or URL to the dataset.
        _type : int
            Type of dataset: 0 or 1 for local/predefined, 2 for URL.

        Returns
        -------
        int
            Internal ID assigned to the dataset.

        Raises
        ------
        ValueError
            If _type is not 0, 1, or 2.
        """
        if (_type == 0) | (_type == 1):  # Local o predefinided
            data = self._load_data_local_or_predefined(file_path)
        elif _type == 2: # URL
            data = self._load_data_url(file_path)
        else:
            raise ValueError("Tipo debe ser 0 (local), 1 (predefinido) o 2 (URL)")
        
        self._actual_id += 1
        return self._actual_id - 1

    def delete(self, id):
        """
        Deletes a dataset by its ID.

        Parameters
        ----------
        id : int
            Internal dataset ID to delete.

        Returns
        -------
        bool
            True if dataset was deleted, False if ID not found.
        """
        if id in self.data:
            del self.data[id]
            self._restructure_ids()
            return True
        return False

    def _restructure_ids(self):
        """
        Reassigns dataset IDs sequentially after deletions to maintain consistency.
        """
        new_data = {}
        for new_id, key in enumerate(sorted(self.data.keys()), 1):
            dataset = self.data[key]
            dataset.id = new_id
            new_data[new_id] = dataset
        self.data = new_data

    def get_data(self):
        """
        Retrieves all registered datasets.

        Returns
        -------
        Dict[int, DatasetInfo]
            Dictionary of dataset ID to DatasetInfo.
        """
        return self.data

    def get_split_dataset(self):
        """
        Retrieves all partitioned dataset groups.

        Returns
        -------
        Dict[str, PartitionGroup]
            Dictionary of partition group key to PartitionGroup.
        """
        return self.partitioned_groups
    
    def get_all_datasets(self):
        """
        Returns all datasets, grouping partitioned datasets together.

        Returns
        -------
        Dict[int, Union[DatasetInfo, dict]]
            Dictionary mapping an index to either DatasetInfo (complete datasets)
            or dict with keys 'train', 'val', 'test' for partitioned groups.
        """
        result = {}
        used_ids = set()
        idx = 0
        values = self.data.values()

        if self.partitioned_groups:
            for group in self.partitioned_groups.values():
                for pid in (group.train_id, group.val_id, group.test_id):
                    if pid is not None:
                        used_ids.add(pid)

        for ds_info in values:
            if ds_info.id not in used_ids:
                result[idx] = ds_info
                idx += 1

        if self.partitioned_groups:
            for group in self.partitioned_groups.values():
                group_entry = {}
                for info in values:
                    if group.train_id == info.id:
                        group_entry['train'] = info
                    elif group.val_id == info.id:
                        group_entry['val'] = info
                    elif group.test_id == info.id:
                        group_entry['test'] = info

                if group_entry:  
                    result[idx] = group_entry
                    idx += 1

        return result

    def get_all_datasets_names(self):
        """
        Retrieves names of all datasets and partition groups.

        Returns
        -------
        Tuple[Dict[int, str], Dict[str, List[Optional[str]]]]
            Tuple with:
            - Dictionary mapping dataset IDs to names for complete datasets.
            - Dictionary mapping partition group IDs to list of train, val, test dataset names.
        """
        datasets = {}
        split_datasets = {}

        used_ids = set()
        for group in self.partitioned_groups.values():
            for pid in (group.train_id, group.val_id, group.test_id):
                if pid is not None:
                    used_ids.add(pid)

        for ds_id, ds_info in self.data.items():
            if ds_id not in used_ids:
                datasets[ds_id] = ds_info.name

        for group_id, group in self.partitioned_groups.items():
            split_datasets[group_id] = [
                self.data[group.train_id].name if group.train_id else None,
                self.data[group.val_id].name if group.val_id else None,
                self.data[group.test_id].name if group.test_id else None
            ]

        return datasets, split_datasets
    
    def set_split_data(self, data):   
        """
        Sets partition groups based on mapping dataset names to dataset IDs.

        Parameters
        ----------
        data : dict
            Dictionary where keys are group IDs and values are lists [train_name, val_name, test_name].
        """
        name_to_id = {info.name: info.id for info in self.data.values()}

        for group_id, group_data in data.items():
            self.partitioned_groups[group_id] = PartitionGroup(
                id=uuid.uuid4(),
                name=f"Group_{group_id}",
                train_id=name_to_id.get(group_data.get("train")),
                val_id=name_to_id.get(group_data.get("val")),
                test_id=name_to_id.get(group_data.get("test")),
            )

    def get_columns (self, id): 
        """
        Gets the list of columns for a dataset by ID.

        Parameters
        ----------
        id : int
            Dataset ID.

        Returns
        -------
        list of str or None
            List of column names if dataset exists, otherwise None.
        """
        if id in self.data:
            return list(self.data[id].data.columns)
        else:
            raise ValueError(f"ID de dataset {id} no encontrado.")
        
    def set_options_x_col (self, options_x_col): 
        """
        Sets the options for feature columns.

        Parameters
        ----------
        options_x_col : dict
            Dictionary of feature columns configuration.
        """
        self.options_x_col = options_x_col

    def get_options_x_col(self, id):
        """
        Gets the feature columns options for a dataset by ID.

        Parameters
        ----------
        id : int
            Dataset ID.

        Returns
        -------
        dict or None
            Feature columns dictionary or None if not found.
        """
        if id in self.data:
            return self.data[id].x_col
        else:
            return {}

    def set_y_col (self, y_col):
        """
        Sets the target column name.

        Parameters
        ----------
        y_col : str
            Name of the target column.
        """
        self.y_col = y_col
    
    def get_y_col(self, id):
        """
        Gets the target column name for a dataset by ID.

        Parameters
        ----------
        id : int
            Dataset ID.

        Returns
        -------
        str or None
            Target column name or None if not found.
        """
        if id in self.data:
            return self.data[id].y_col
        else:
            return None
    
    def add_x_col (self, id, options_x_col):
        """
        Assigns feature columns to a dataset.

        Parameters
        ----------
        id : int
            Dataset ID.
        options_x_col : dict
            Dictionary of feature columns configuration.
        """
        if id not in self.data:
            raise ValueError(f"Dataset con ID {id} no encontrado.")

        dataset = self.data[id]
        dataset.x_col = options_x_col

    def add_y_col (self, id, y_col):
        """
        Assigns target column to a dataset.

        Parameters
        ----------
        id : int
            Dataset ID.
        y_col : str
            Name of the target column.
        """
        if id not in self.data:
            raise ValueError(f"Dataset con ID {id} no encontrado.")

        dataset = self.data[id]
        dataset.y_col = y_col