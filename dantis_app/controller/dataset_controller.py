
import requests

import os
import pandas as pd
import io
from scipy.io import arff
import arff as liac_arff 
from dataclasses import dataclass, field
import uuid
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class DatasetInfo:
    id: int
    name: str
    path: str
    data: pd.DataFrame
    x_col: dict = field(default_factory=dict)
    y_col: str = None    

@dataclass
class PartitionGroup:
    id: str  # UUID
    name: str
    train_id: Optional[int] = None
    val_id: Optional[int] = None
    test_id: Optional[int] = None

COMMON_ENCODINGS = ["utf-8", "latin1", "ISO-8859-1", "utf-16"]
COMMON_SEPARATORS = [",", ";", "\t"]

def _try_read_csv(content, from_file=True):
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
        if not is_remote:
            data, meta = arff.loadarff(path_or_url)
        else:
            text = response.content.decode("utf-8", errors="ignore")
            data, meta = arff.loadarff(io.StringIO(text))
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Formato no soportado: {ext}")

def load_predefined_datasets_name(github_repo_url):
    """Carga la lista de datasets disponibles desde GitHub."""
    try:
        # Solicitar contenido de la carpeta
        response = requests.get(github_repo_url)

        # Verificar si la respuesta fue exitosa
        if response.status_code == 200:
            datasets = response.json()  # Convertir respuesta JSON
            datasets_list = []  # Lista para almacenar los nombres de archivos

            for file in datasets:
                if file["name"].endswith(".csv"):  # Solo incluir archivos CSV
                    datasets_list.append(file["name"])

            return datasets_list
        else:
            print(f"Error al obtener datasets: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error al procesar la solicitud: {e}")
        return []

class DatasetController:
    def __init__(self):
        self.datos: Dict[int, DatasetInfo] = {}
        self.partitioned_groups: Dict[str, PartitionGroup] = {}
        self._actual_id = 1   
        self.options_x_col = {}
        self.y_col = None

    def _register_dataset(self, name: str, path: str, data: pd.DataFrame):
        if self.datos: 
            ultimo_id = max(self.datos.keys())
            self._actual_id = ultimo_id + 1
        info = DatasetInfo(id=self._actual_id, name=name, path=path, data=data)
        self.datos[self._actual_id] = info
        return info
    
    def add_patitioned_group(self):
        pass

    def _load_data_local_o_predefinido(self, file_path:str):
        data = _load_file(file_path, is_remote=False)
        return self._register_dataset(name=os.path.basename(file_path), path=file_path, data=data)

    def _load_data_url(self, url: str):
        data = _load_file(url, is_remote=True)
        return self._register_dataset(name=os.path.basename(url), path=url, data=data)

    def add_datasets(self, file_path, tipo):
        if (tipo == 0) | (tipo == 1):  # Local o predefinido
            data = self._load_data_local_o_predefinido(file_path)
        elif tipo == 2: # URL
            data = self._load_data_url(file_path)
        else:
            raise ValueError("Tipo debe ser 0 (local), 1 (predefinido) o 2 (URL)")
        
        self._actual_id += 1
        return self._actual_id - 1

    def eliminar(self, id):
        if id in self.datos:
            del self.datos[id]
            self._reestructurar_ids()
            return True
        return False

    def _reestructurar_ids(self):
        nuevo_datos = {}
        for nuevo_id, key in enumerate(sorted(self.datos.keys()), 1):
            dataset = self.datos[key]
            dataset.id = nuevo_id  # Actualiza el id interno del objeto
            nuevo_datos[nuevo_id] = dataset
        self.datos = nuevo_datos

    def get_data(self):
        return self.datos
    
    def get_split_dataset(self):
        return self.partitioned_groups
    
    def get_all_datasets(self):
        result = {}
        used_ids = set()
        idx = 0

        # 1. Identificar los IDs usados en particiones
        for group in self.partitioned_groups.values():
            for pid in (group.train_id, group.val_id, group.test_id):
                if pid is not None:
                    used_ids.add(pid)

        # 2. Agregar datasets NO usados en particiones (completos)
        for ds_id, ds_info in self.datos.items():
            if ds_id not in used_ids:
                result[idx] = ds_info
                idx += 1

        # 3. Agregar datasets PARTICIONADOS agrupados por grupo
        for group in self.partitioned_groups.values():
            group_entry = {}
            if group.train_id in self.datos:
                group_entry['train'] = self.datos[group.train_id]
            if group.val_id in self.datos:
                group_entry['val'] = self.datos[group.val_id]
            if group.test_id in self.datos:
                group_entry['test'] = self.datos[group.test_id]
            result[idx] = group_entry
            idx += 1

        return result


    def get_all_datasets_names(self):
        datasets = {}
        split_datasets = {}

        # 1. Identificar todos los IDs usados en particiones
        used_ids = set()
        for group in self.partitioned_groups.values():
            for pid in (group.train_id, group.val_id, group.test_id):
                if pid is not None:
                    used_ids.add(pid)

        # 2. Agregar datasets completos (no usados en particiones)
        for ds_id, ds_info in self.datos.items():
            if ds_id not in used_ids:
                datasets[ds_id] = ds_info.name

        # 3. Agregar datasets particionados por grupo
        for group_id, group in self.partitioned_groups.items():
            split_datasets[group_id] = [
                self.datos[group.train_id].name if group.train_id else None,
                self.datos[group.val_id].name if group.val_id else None,
                self.datos[group.test_id].name if group.test_id else None
            ]

        return datasets, split_datasets
    
    def set_split_data(self, data):   

        name_to_id = {info.name: dataset_id for dataset_id, info in self.datos.items()}

        for group_id, group_data in data.items():            
            self.partitioned_groups[group_id] = PartitionGroup(
                id=str(uuid.uuid4()),
                name=f"Group_{group_id}",
                train_id=name_to_id.get(group_data.get("train")),
                val_id=name_to_id.get(group_data.get("val")),
                test_id=name_to_id.get(group_data.get("test")),
            ) 

    def get_columns (self, id): 
        if id in self.datos:
            return list(self.datos[id].data.columns)
        else:
            raise ValueError(f"ID de dataset {id} no encontrado.")
        
    def set_options_x_col (self, options_x_col): 
        self.options_x_col = options_x_col

    def get_options_x_col(self, id):
        if id in self.datos:
            return self.datos[id].x_col
        else:
            return {}

    def set_y_col (self, y_col):
        self.y_col = y_col
    
    def get_y_col(self, id):
        if id in self.datos:
            return self.datos[id].y_col
        else:
            return None
    
    def add_x_col (self, id, options_x_col):
        if id not in self.datos:
            raise ValueError(f"Dataset con ID {id} no encontrado.")
        
        dataset = self.datos[id]
        dataset.x_col = options_x_col

    def add_y_col (self, id, y_col):
        if id not in self.datos:
            raise ValueError(f"Dataset con ID {id} no encontrado.")
        
        dataset = self.datos[id]
        dataset.y_col = y_col


if __name__ == "__main__":
    def test_datasets():
        controller = DatasetController()

        print("Probando CSV remoto...")
        ds_csv = controller.add_url("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
        print(f"CSV ({ds_csv.name}) - shape: {ds_csv.data.shape}")

        print("\nProbando JSON remoto...")
        ds_json = controller.add_url("https://raw.githubusercontent.com/vega/vega-datasets/master/data/cars.json")
        print(f"JSON ({ds_json.name}) - shape: {ds_json.data.shape}")

        # print("\nProbando Excel remoto (.xlsx)...")
        # ds_xlsx = controller.add_url("subir_alguno.xlsx")  
        # print(f"Excel ({ds_xlsx.name}) - shape: {ds_xlsx.data.shape}")

        print("\nProbando ARFF url...")
        ds_arff = controller.add_url("https://raw.githubusercontent.com/renatopp/arff-datasets/refs/heads/master/classification/diabetes.arff")  # necesitas tenerlo descargado
        print(f"ARFF ({ds_arff.name}) - shape: {ds_arff.data.shape}")

    test_datasets()