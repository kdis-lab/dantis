import importlib
import pkgutil
import inspect
import numpy as np

# --------------------------------------------
# TODO: Temporal fix to allow imports from parent directory 
# This should be removed when the package is public.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# --------------------------------------------

import dantis
from dantis.algorithmbase import AlgorithmBase


def _serialize_value(value):
    """Serialize a value to make it JSON-compatible."""
    if value is inspect._empty:
        return "empty"
    elif isinstance(value, type):
        return value.__name__
    elif callable(value):
        return value.__name__
    elif hasattr(value, "__class__") and hasattr(value, "get_params"):
        return {
            "__model__": value.__class__.__name__,
            "__module__": value.__class__.__module__,
            "params": {k: _serialize_value(v) for k, v in value.get_params().items()}
        }
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _serialize_hyperparameters(hyperparameters):
    """Recursively serialize a dictionary of hyperparameters."""
    return {k: _serialize_value(v) for k, v in hyperparameters.items()}


def discover_model_classes(package):
    """
    Discover all classes in a given package that inherit from AlgorithmBase.

    This function recursively walks through all modules in the specified package,
    imports them, and inspects their contents to find classes that are subclasses
    of AlgorithmBase (excluding AlgorithmBase itself). The discovered classes are
    grouped by model type, which is inferred from the module path.

    Parameters
    ----------
    package : module
        The Python package to search for model classes. Must have a __path__ attribute.

        A dictionary mapping model type (str) to a list of class references (type).
        The model type is determined by the second part of the module's dotted path,
        or set to 'classical' if not available.

    Notes
    -----
    - Only classes defined in their respective modules (not imported) are considered.
    - If a module cannot be imported, an error message is printed and the module is skipped.
    """

    discovered = {}

    for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            module = importlib.import_module(modname)
            classes = inspect.getmembers(module, inspect.isclass)

            valid_classes = [
                cls for name, cls in classes
                if issubclass(cls, AlgorithmBase) and cls != AlgorithmBase and cls.__module__ == modname
            ]

            if valid_classes:
                parts = modname.split(".")
                model_type = parts[-2] # if len(parts) > 2 else "classical"

                discovered.setdefault(model_type, []).extend(valid_classes)

        except Exception as e:
            print(f"[ERROR] No se pudo importar {modname}: {e}")

    return discovered


def extract_model_names_by_type(package):
    """
    Retorna nombres de modelos organizados por tipo de algoritmo (basado en módulo).
    """
    classes_by_type = discover_model_classes(package)
    print(f"[INFO] Descubiertas {len(classes_by_type)} tipos de modelos en el paquete '{package.__name__}'.")
    print(classes_by_type.keys())
    return {
        model_type: [cls.__name__ for cls in class_list if cls.__name__ != "Topology"]
        for model_type, class_list in classes_by_type.items()
    }


def extract_default_hyperparameters(package):
    """
    Instancia cada clase y extrae sus hiperparámetros por defecto.
    """
    params_by_model = {}

    for model_type, class_list in discover_model_classes(package).items():
        for cls in class_list:
            if cls.__name__ == "Topology":
                continue

            if hasattr(cls, "get_default_hyperparameters"):
                hparams = cls.get_default_hyperparameters()
            else:
                instance = cls({})
                hparams = instance._hyperparameter

            params_by_model[cls.__name__] = _serialize_hyperparameters(hparams)
            
            """
            try:
                if hasattr(cls, "get_default_hyperparameters"):
                    hparams = cls.get_default_hyperparameters()
                else:
                    instance = cls({})
                    hparams = instance._hyperparameter

                params_by_model[cls.__name__] = _serialize_hyperparameters(hparams)
            except Exception as err:
                print(f"[WARN] No se pudo instanciar '{cls.__name__}': {err}")
            """
    return params_by_model

def instantiate_model_by_name(class_name, config=None, x=None, y=None, x_test=None, y_test=None):
    package = dantis
    config = config or {}
    x = x if x is not None else np.array([1, 2])
    y = y if y is not None else np.array([1, 2])
    x_test = x_test if x_test is not None else None
    y_test = y_test if y_test is not None else None

    for _, modname, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            module = importlib.import_module(modname)
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if name == class_name and cls.__module__ == modname:
                    # Instanciar pasando test si la clase lo acepta
                    # Comprobamos si el constructor acepta esos parámetros:
                    params = inspect.signature(cls.__init__).parameters
                    return cls(config)
        except Exception as e:
            print(f"[ERROR] No se pudo importar {modname}: {e}")

    print(f"[WARN] Clase '{class_name}' no encontrada en el paquete.")
    return None

# Prueba
if __name__ == "__main__":
    result = extract_model_names_by_type(dantis)
    for model_type, classes in result.items():
        print(f"\n{model_type}")
        for cls_name in classes:
            print(f"  - {cls_name}")

    params_by_model = extract_default_hyperparameters(dantis)
    for model, params in params_by_model.items():
        print(f"\n{model}:")
        for k, v in params.items():
            print(f"  {k}: {v}")

