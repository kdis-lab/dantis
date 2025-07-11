# Guía para Incorporar Nuevos Algoritmos a la Librería de Detección de Anomalías

Esta guía describe cómo integrar un nuevo algoritmo a la librería del proyecto Granada, basada en detección de anomalías en series temporales. Todo nuevo algoritmo debe heredar de la clase abstracta `AlgorithmBase`.

## 1. Estructura General

Cada algoritmo debe definirse en un archivo propio dentro del módulo correspondiente, por ejemplo: `my_new_algo.py`.

## 2. Herencia de `AlgorithmBase`

La clase del nuevo algoritmo debe heredar de `AlgorithmBase`:

```python
from .. import algorithmbase
from .. import utils
from pyod.models import my_new_algo

class MyNewAlgo(algorithmbase.AlgorithmBase):
```

## 3. Métodos Obligatorios

### `__init__(self, hyperparameter: dict)`

Debe:

- Llamar a `super().__init__()`.
- Llamar a `_create_model()` para instanciar el modelo con los hiperparámetros.

### `fit(self, x_train: np.ndarray, y_train: np.ndarray = None)`

Debe entrenar el modelo con los datos de entrada.

### `decision_function(self, x: np.ndarray) -> np.ndarray`

Devuelve los scores continuos de anormalidad.

### `predict(self, x: np.ndarray) -> np.ndarray`

Devuelve la predicción binaria. Puede implementar este método como `return self.get_anomaly_score(x)` si no se requiere lógica adicional.

## 4. Gestión de Hiperparámetros

Utiliza `utils.ParameterManagement` para validar y completar los hiperparámetros. Esto asegura que los tipos y valores sean correctos.

```python
def set_hyperparameter(self, hyperparameter):
    self._hyperparameter = hyperparameter
    self._model = my_new_algo.MyNewAlgo
    parameters_management = utils.ParameterManagement(self._model.__init__)
    hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
    self._hyperparameter = parameters_management.complete_parameters(hyperparameter)
```

## 5. Creación del Modelo

Método interno que instancia el modelo con los hiperparámetros.

```python
def _create_model(self):
    self.set_hyperparameter(self._hyperparameter)
    self._model = my_new_algo.MyNewAlgo(**self._hyperparameter)
```

## 6. Guardado y Carga del Modelo

Por defecto, estos métodos ya están implementados en `AlgorithmBase`:

```python
def save_model(self, path_model):
    super().save_model(path_model)

def load_model(self, path_model):
    super().load_model(path_model)
```

## 7. Ejemplo Mínimo

```python
from pyod.models import my_new_algo
from .. import algorithmbase
from .. import utils

class MyNewAlgo(algorithmbase.AlgorithmBase):

    def __init__(self, hyperparameter: dict):
        super().__init__(hyperparameter)
        self._create_model()

    def decision_function(self, x):
        return self._model.decision_function(x)

    def fit(self, x_train, y_train=None):
        self._model.fit(x_train)
        return self.get_anomaly_score(x_train)

    def predict(self, x):
        return self.get_anomaly_score(x)

    def set_hyperparameter(self, hyperparameter):
        self._hyperparameter = hyperparameter
        self._model = my_new_algo.MyNewAlgo
        params = utils.ParameterManagement(self._model.__init__)
        hyperparameter = params.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = params.complete_parameters(hyperparameter)

    def _create_model(self):
        self.set_hyperparameter(self._hyperparameter)
        self._model = my_new_algo.MyNewAlgo(**self._hyperparameter)
```

## 8. Recomendaciones

- No modificar directamente `_model` desde fuera de `set_hyperparameter` o `_create_model`.
- Utilizar la extensión `.joblib` para guardar y cargar modelos.
- Evitar usar `y_train` en algoritmos no supervisados, a menos que sea estrictamente necesario.