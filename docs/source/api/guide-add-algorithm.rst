Guide for Adding New Algorithms
===============================

This guide describes how to integrate a new algorithm into the **Granada project's anomaly detection library**, which is focused on time series data.  
Each new algorithm must inherit from the abstract class ``AlgorithmBase``.

1. General Structure
--------------------

Each algorithm should be defined in its own file within the appropriate module, for example: ``my_new_algo.py``.

2. Inheriting from ``AlgorithmBase``
------------------------------------

The algorithm class must inherit from ``AlgorithmBase``:

.. code-block:: python

    from .. import algorithmbase
    from .. import utils
    from pyod.models import my_new_algo

    class MyNewAlgo(algorithmbase.AlgorithmBase):
        ...

3. Required Methods
-------------------

**__init__(self, hyperparameter: dict)**

- Call ``super().__init__()``.
- Call ``_create_model()`` to instantiate the model using the given hyperparameters.

**fit(self, x_train: np.ndarray, y_train: np.ndarray = None)**

- Should train the model with the input data.

**decision_function(self, x: np.ndarray) -> np.ndarray**

- Returns continuous anomaly scores.

**predict(self, x: np.ndarray) -> np.ndarray**

- Returns binary predictions. You can simply return ``self.get_anomaly_score(x)`` if no additional logic is needed.

4. Hyperparameter Management
----------------------------

Use ``utils.ParameterManagement`` to validate and complete the hyperparameters. This ensures correct types and values.

.. code-block:: python

    def set_hyperparameter(self, hyperparameter):
        self._hyperparameter = hyperparameter
        self._model = my_new_algo.MyNewAlgo
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

5. Model Creation
-----------------

An internal method that instantiates the model with the configured hyperparameters.

.. code-block:: python

    def _create_model(self):
        self.set_hyperparameter(self._hyperparameter)
        self._model = my_new_algo.MyNewAlgo(**self._hyperparameter)

6. Saving and Loading the Model
-------------------------------

These methods are already implemented in ``AlgorithmBase`` and usually don't need to be overridden:

.. code-block:: python

    def save_model(self, path_model):
        super().save_model(path_model)

    def load_model(self, path_model):
        super().load_model(path_model)

7. Minimal Example
------------------

.. code-block:: python

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

8. Recommendations
------------------

- Avoid modifying ``_model`` directly outside of ``set_hyperparameter`` or ``_create_model``.
- Use ``.joblib`` as the file extension for saving and loading models.
- Avoid using ``y_train`` in unsupervised algorithms unless explicitly required.
