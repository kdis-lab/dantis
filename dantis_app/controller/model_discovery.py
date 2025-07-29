import importlib
import pkgutil
import inspect
import numpy as np
import logging

# --------------------------------------------
# TODO: Temporal fix to allow imports from parent directory 
# This should be removed when the package is public.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import dantis
from dantis.algorithmbase import AlgorithmBase
# --------------------------------------------

def _serialize_value(value):
    """
    Serializes a Python object into a JSON-compatible format.

    This function converts various types of objects—such as classes, callables, model instances,
    lists, dictionaries, and numpy arrays—into formats suitable for JSON serialization.
    Useful for exporting model configurations or hyperparameters.

    Parameters
    ----------
    value : any
        The object to be serialized.

    Returns
    -------
    any
        A JSON-compatible representation of the input object.
    """
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
    """
    Recursively serializes a dictionary of hyperparameters.

    Each value in the hyperparameter dictionary is passed through `_serialize_value` 
    to ensure the final dictionary is fully JSON-compatible.

    Parameters
    ----------
    hyperparameters : dict
        Dictionary of hyperparameters to be serialized.

    Returns
    -------
    dict
        A JSON-compatible dictionary of hyperparameters.
    """
    return {k: _serialize_value(v) for k, v in hyperparameters.items()}

def discover_model_classes(package):
    """
    Discover all model classes in a given package that inherit from AlgorithmBase.

    This function recursively inspects all modules in the specified package and identifies
    classes that:
    - Are subclasses of `AlgorithmBase`
    - Are defined in the same module (not imported)
    - Are not the `AlgorithmBase` class itself

    The discovered classes are categorized by their model type, inferred from the module name.

    Parameters
    ----------
    package : module
        The Python package to search. Must have a `__path__` attribute (e.g., a package, not a module).

    Returns
    -------
    dict[str, list[type]]
        A dictionary mapping model type names to lists of class objects.

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
            logging.error(f"No se pudo importar {modname}: {e}")

    return discovered

def extract_model_names_by_type(package):
    """
    Extracts model class names grouped by algorithm type.

    Based on the discovery of subclasses of `AlgorithmBase`, this function returns only
    the names (strings) of valid model classes, excluding utility classes like "Topology".

    Parameters
    ----------
    package : module
        The package to search for model classes (e.g., `dantis`).

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping model types to lists of class names.
    """
    classes_by_type = discover_model_classes(package)
    return {
        model_type: [cls.__name__ for cls in class_list if cls.__name__ != "Topology"]
        for model_type, class_list in classes_by_type.items()
    }

def extract_default_hyperparameters(package):
    """
    Extracts default hyperparameters for each model class in a package.

    For each class that inherits from `AlgorithmBase`, this function either:
    - Uses the class method `get_default_hyperparameters()`, if available.
    - Instantiates the class with an empty config and reads the `_hyperparameter` attribute.

    Parameters
    ----------
    package : module
        The package to inspect for model classes.

    Returns
    -------
    dict[str, dict]
        A dictionary mapping class names to their serialized default hyperparameters.
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
            
    return params_by_model

def instantiate_model_by_name(class_name, config=None, x=None, y=None, x_test=None, y_test=None):
    """
    Instantiates a model class by its name from a given package.

    Searches the given package recursively for a class that matches `class_name`
    and is defined in its own module. Upon finding it, instantiates the class with 
    the given configuration.

    If input data (`x`, `y`, etc.) is not provided, defaults are used to avoid errors
    during construction, though the data is not passed into the constructor.

    Parameters
    ----------
    class_name : str
        Name of the class to instantiate.

    config : dict, optional
        Dictionary of hyperparameters to configure the model.

    x : array-like, optional
        Feature data. Default is a dummy array.

    y : array-like, optional
        Target data. Default is a dummy array.

    x_test : array-like, optional
        Test features.

    y_test : array-like, optional
        Test targets.

    Returns
    -------
    object or None
        An instance of the model class if found; otherwise, None.
    """
    import dantis
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
                    params = inspect.signature(cls.__init__).parameters
                    return cls(config)
        except Exception as e:
            logging.error(f"No se pudo importar {modname}: {e}")

    logging.warning(f"Clase '{class_name}' no encontrada en el paquete.")
    return None