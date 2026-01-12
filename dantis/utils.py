import inspect

class ParameterManagement:
    """Utility to extract defaults from a callable and validate/complete user
    supplied hyperparameters.

    Behavior:
    - `get_parameter_function` extracts parameters that have explicit defaults
      (skips parameters with inspect._empty).
    - `check_hyperparameter_type` attempts to cast user values to the type of
      the default value when possible (safe casts only).
    - `complete_parameters` returns a new dict merging defaults and user
      provided values (does not mutate the stored defaults).
    """

    def __init__(self, function):
        self._default_parameters = self.get_parameter_function(function)

    def __get_parameter_correct_type(self, param_key, param_value):
        """Attempt to coerce `param_value` to the type of the default for
        `param_key`. If the default is `None` or coercion is unsafe, the
        original value is returned.
        """
        if param_key not in self._default_parameters:
            return param_value

        default = self._default_parameters[param_key]
        # If default is None we cannot infer a type -> leave as-is
        if default is None:
            return param_value

        real_type = type(default)

        # Handle numpy scalar defaults
        try:
            import numpy as _np

            if isinstance(default, (_np.integer, _np.floating)):
                real_type = type(default)
        except Exception:
            pass

        # Already correct type
        if isinstance(param_value, real_type):
            return param_value

        # Try safe casting for common builtins
        try:
            if real_type is bool:
                if isinstance(param_value, str):
                    return param_value.lower() in ("true", "1", "yes")
                return bool(param_value)
            if real_type in (int, float, str):
                return real_type(param_value)
            if real_type is dict:
                if isinstance(param_value, str):
                    import json

                    return json.loads(param_value)
                return dict(param_value)
            if real_type is list:
                if isinstance(param_value, str):
                    import json

                    return json.loads(param_value)
                return list(param_value)
            if real_type is tuple:
                if isinstance(param_value, str):
                    import json

                    return tuple(json.loads(param_value))
                return tuple(param_value)
            if real_type is set:
                if isinstance(param_value, str):
                    import json

                    return set(json.loads(param_value))
                return set(param_value)

            # Fallback: attempt a direct cast
            return real_type(param_value)
        except Exception as e:
            raise Exception(f"Error, type for '{param_key}' is not correct: {e}")

    def check_hyperparameter_type(self, user_parameters: dict) -> dict:
        """Return a new dict with user parameters coerced to the types of the
        defaults where possible. Unknown keys are passed through unchanged.
        """
        if not user_parameters:
            return {}

        out = {}
        for key, val in user_parameters.items():
            if key not in self._default_parameters:
                out[key] = val
                continue
            out[key] = self.__get_parameter_correct_type(key, val)

        return out

    @staticmethod
    def get_parameter_function(function) -> dict:
        """Extract parameters with defaults from a callable's signature.

        Returns a dict mapping parameter name -> default value. Parameters with
        no default (inspect._empty) are skipped.
        """
        if callable(function):
            signature = inspect.signature(function)
            parameters = signature.parameters
            args = {
                name: param.default
                for name, param in parameters.items()
                if name not in ("self", "args", "kwargs") and param.default is not inspect._empty
            }
            return args
        return {}

    def complete_parameters(self, user_parameters: dict) -> dict:
        """Return a new dict where missing parameters are filled from defaults.

        The stored `_default_parameters` is not mutated.
        """
        merged = {}
        for k, default in self._default_parameters.items():
            if user_parameters and k in user_parameters:
                merged[k] = user_parameters[k]
            else:
                merged[k] = default
        return merged


class SupervisedInputDataError(ValueError):
    def __init__(self, mensaje="En algoritmos supervisados es necesario proporcionar los datos de entrada y las etiquetas a predecir"):
        self.mensaje = mensaje
        super().__init__(self.mensaje)
