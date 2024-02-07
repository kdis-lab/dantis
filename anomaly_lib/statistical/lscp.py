from pyod.models import lscp # lscp.LSCP
import numpy as np

from .. import algorithmbase
from .. import utils


class LSCP(algorithmbase.AlgorithmBase):

    def __init__(self, hyperparameter: dict, x_train: np.array, y_train: np.array = None, x_test: np.array = None,
                 y_test: np.array = None):
        """
            LSCP class constructor.

            Args:
                - hyperparameters (dict): a dictionary containing the hyperparameters
                  for the LSCP algorithm.
                - x_train (np.array): Characteristics of the training data.
                - x_test (np.array): Characteristics of the test data.
                - y_train (np.array): Training data labels.
                - y_test (np.array): Tags of the test data.
        """

        if "detector_list" not in hyperparameter.keys():
            raise utils.SupervisedInputDataError("detector_list param is needed")
        
        if not hasattr(hyperparameter["detector_list"], "__iter__"):
            raise utils.SupervisedInputDataError("LSCP needs a list with the ad_algorithms to be used")

        super().__init__(x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test,
                         hyperparameter=hyperparameter)
        
        self._create_model()

    def fit(self):
        """
            Fit detector. Adjusts the LSCP model using the training data.

            Returns:
                 The outlier scores of the training data [0,1].

        """
        self._model.fit(self._x_train)

        return self._model.decision_scores_

    def predict(self):
        """
            Predict whether a particular sample is an outlier or not.
            Adjusts the LSCP model using the test data.

            Returns:
                Predict probability of outlier
        """
        x_test = super().get_x_test()

        return self._model.predict_proba(x_test)

    def save_model(self, path_model):
        """
            Saves the LSCP model to the specified file.

            Args:
                - path_model (str): Path of the file where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
            Loads the LSCP model from the specified file.

            Args:
                - path_model (str): Path of the file where the model is located.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
            Sets the hyperparameters of the model.

            Args:
                - hyperparameter (dict): A dictionary containing the hyperparameters of the model.
        """
        self._hyperparameter = hyperparameter
        self._model = lscp.LSCP
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
            Creates the LSCP model with the provided hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = lscp.LSCP(**self._hyperparameter)