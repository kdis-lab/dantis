import stumpy  # stumpy.stump & stumpy.mstump
import numpy as np

from .. import algorithmbase
from .. import utils


class MatrixProfile(algorithmbase.AlgorithmBase):

    def __init__(self, hyperparameter: dict, x_train: np.array, y_train: np.array = None, x_test: np.array = None,
                 y_test: np.array = None):
        """
            Matrix Profile constructor..

            Args:
                - hyperparametros (dict): A dict with the hyperparams for the Matrix Profile algorithm.
                - x_train (np.array): It is maintained for reasons of consistency with the other models.
                - x_test (np.array): It is maintained for reasons of consistency with the other models.
                - y_train (np.array): It is maintained for reasons of consistency with the other models.
                - y_test (np.array): It is maintained for reasons of consistency with the other models.
        """
        super().__init__(x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test,
                         hyperparameter=hyperparameter)
        
        self._create_model()

    def fit(self):
        """
            Fit detector. Adjusts the Matrix Profile model using the training data.

            Returns:
                 The outlier scores of the training data.
        """
        x_train = self.get_x_train()
        hyperparameter = self.get_hyperparameter()
        if x_train.shape[0] > 1:
            hyperparameter["T"] = x_train
        else:
            hyperparameter["T_A"] = x_train.flatten()
        hyperparameter = {clave: valor for clave, valor in hyperparameter.items() if clave != 'threshold'}

        matrix_profile = self._model(**hyperparameter)
        return matrix_profile

    def predict(self):
        """
            Predict whether a particular sample is an outlier or not.
            Adjusts the Matrix Profile model using the test data.

            Returns:
                Predict probability of outlier
        """
        x_test = super().get_x_test()
        hyperparameter = self.get_hyperparameter()
        if x_test.shape[0] > 1:
            hyperparameter["T"] = x_test
        else:
            hyperparameter["T_A"] = x_test.flatten()

        hyperparameter_copy = {clave: valor for clave, valor in hyperparameter.items() if clave != 'threshold'}

        matrix_profile = self._model(**hyperparameter_copy)
        threshold = hyperparameter["threshold"]
        if type(matrix_profile) is tuple:
            matrix_profile = matrix_profile[0]
        anomaly_index = np.where(matrix_profile[:, 0] > threshold)[0]
        return anomaly_index

    def save_model(self, path_model):
        """
            This model can not be saved
        """
        print("Warning: This model cannot be saved")

    def load_model(self, path_model):
        """
             This model cannot be loaded
         """
        print("Warning: This model cannot be loaded")

    def set_hyperparameter(self, hyperparameter):
        """
            Sets the hyperparameters of the model.

            Args:
                - hyperparameter (dict): A dictionary containing the hyperparameters of the model.
        """
        self._hyperparameter = hyperparameter
        selected_model = stumpy.mstump if self._x_train.shape[0] > 1 else stumpy.stump
        threshold = hyperparameter["threshold"]
        self._model = selected_model
        parameters_management = utils.ParameterManagement(self._model)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)
        self._hyperparameter["threshold"] = threshold

    def _create_model(self):
        """
            Creates the Matrix Profile model with the provided hyperparameters.
        """

        if self._x_train.shape[0] > 1:
            self._hyperparameter["T"] = self._x_train
        else:
            self._hyperparameter["T_A"] = self._x_train.flatten()

        self.set_hyperparameter(self._hyperparameter)