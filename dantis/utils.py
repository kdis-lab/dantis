import inspect

class ParameterManagement:
    """
    Utility class for managing and validating hyperparameters of functions or models.

    This class extracts default parameter values from a callable (typically a model constructor),
    checks user-defined parameters against expected types, and fills in missing parameters 
    with their default values.

    Attributes
    ----------
    _default_parameters : dict
        Dictionary containing default parameter names and values from the target function.
    _types : dict
        Dictionary mapping type names (as strings) to actual Python type objects.

    Parameters
    ----------
    function : callable
        The target function (usually a model's constructor) whose parameters are to be managed.

    Methods
    -------
    get_parameter_function(function)
        Extracts parameters and their default values from the function signature.
    
    check_hyperparameter_type(user_parameters)
        Validates the types of user-supplied parameters, attempting casting if necessary.
    
    complete_parameters(user_parameters)
        Completes the user-supplied dictionary with missing parameters from defaults.
    """

    _default_parameters = None
    _types = {
        "int": int, "str": str, "float": float, "dict": dict, "list": list,
        "set": set, "tuple": tuple, "None": None, "bool": bool
    }

    def __init__(self, function):
        """
        Initialize the ParameterManagement with a target function.

        Parameters
        ----------
        function : callable
            A function (typically a model's __init__) whose default parameters are extracted.
        """
        self._default_parameters = self.get_parameter_function(function)

    def __get_parameter_correct_type(self, param_key, param_value):
        """
        Attempt to cast a parameter to its correct type based on defaults.

        Parameters
        ----------
        param_key : str
            Name of the parameter.
        param_value : any
            Value provided by the user.

        Returns
        -------
        any
            The value cast to the correct type.

        Raises
        ------
        Exception
            If type casting fails or if the parameter has an unexpected type.
        """
        type_param = type(param_value)
        real_type = type(self._default_parameters[param_key])
        if type_param == real_type or real_type is None or real_type is type:
            return param_value
        try:
            type_to_cast = self._types[type_param.__name__]
            param_value = type_to_cast(param_value)
        except Exception:
            raise Exception(f"Error, type for '{param_key}' is not correct")

        return param_value

    def check_hyperparameter_type(self, user_parameters: dict) -> dict:
        """
        Check and enforce correct types for user-provided hyperparameters.

        Parameters
        ----------
        user_parameters : dict
            Dictionary containing user-defined hyperparameters.

        Returns
        -------
        dict
            Hyperparameter dictionary with values cast to their correct types (if necessary).
        """
        for param_key in user_parameters.keys():
            if param_key not in self._default_parameters.keys():
                continue
            param = self.__get_parameter_correct_type(param_key, user_parameters[param_key])
            user_parameters[param_key] = param

        return user_parameters

    @staticmethod
    def get_parameter_function(function) -> dict:
        """
        Extracts the default parameters from a function or method.

        Parameters
        ----------
        function : callable
            A function or method whose signature is to be inspected.

        Returns
        -------
        dict
            Dictionary of parameter names and their default values.
        """
        if hasattr(function, '__call__'):
            function_signature = inspect.signature(function)
            parameters = function_signature.parameters

            args = {
                name: parameter.default for name, parameter in parameters.items()
                if name not in ["self", "kwargs", "args"]
            }
            return args
        else:
            return {}

    def complete_parameters(self, user_parameters: dict) -> dict:
        """
        Complete missing hyperparameters using the defaults from the target function.

        Parameters
        ----------
        user_parameters : dict
            Dictionary with user-defined parameters.

        Returns
        -------
        dict
            Dictionary containing both user-specified and default hyperparameters.
        """
        non_defined_args = list(set(self._default_parameters.keys() - user_parameters.keys()))
        defined_args = set(self._default_parameters.keys() - non_defined_args)

        for args in defined_args:
            self._default_parameters[args] = user_parameters[args]

        return self._default_parameters


class SupervisedInputDataError(ValueError):
    def __init__(self, mensaje="En algoritmos supervisados es necesario proporcionar los datos de entrada y las etiquetas a predecir"):
        self.mensaje = mensaje
        super().__init__(self.mensaje)
