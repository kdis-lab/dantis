import unittest
import pandas as pd
import os
import numpy as np
import random

import sys
sys.path.append("..")

# Imports pyod
from pyod.models import iforest, ecod, lof, knn, ocsvm
from pyod.models import alad, anogan, auto_encoder, deep_svdd, mo_gaal, so_gaal, vae

from pyod.models import lunar, abod, cblof, cd, copod, feature_bagging, gmm, hbos, inne, kde, kpca, lmdd, loci, loda
from pyod.models import lscp, mad, mcd, pca, qmcd, rgraph, rod, sampling, sod, sos, xgbod


from statsmodels.tsa.arima import model as arima
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.statespace import varmax
from statsmodels.tsa import holtwinters 
from statsmodels.tsa.exponential_smoothing import ets 

import stumpy
#import stumpy

'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
'''
from sklearn import svm, tree, neural_network, naive_bayes, ensemble
from sklearn.preprocessing import MinMaxScaler

from pyod.utils.data import generate_data

from anomaly_lib import statistical, deep_learning, machine_learning
from anomaly_lib import utils




class InitModels(unittest.TestCase):

    def generate_time_series(timesteps: int):
        """
            Generate time series

            Args:
                - timesteps (int): Number of time steps or instants in time for which data are generated in the simulation.

            Returns:
                data (array): Returns the dataset with the time series that have been generated.
        """
        time = np.linspace(0, 10, timesteps)
        data = np.sin(time) + np.random.normal(scale=0.1, size=timesteps)
        return data

    ts = generate_time_series(100).reshape(-1, 1)

    def check_default_model(self, model, function_model):
        """
            Performs unit tests with default values

            Args:
                - model (str): Instance of a model

            Returns:
                function_model (str): Instance of a class
        """

        hyper_default = utils.ParameterManagement(function_model.__init__)._default_parameters

        hyper_model = model.get_hyperparameter()
        

        # Assert de comprobar que son iguales
        self.assertEqual(hyper_model, hyper_default)
        # Assert Entrenar
        try:
            score = model.fit()
        except Exception as e:
            raise Exception("Error al entrenar")

        self.assertEqual(len(score), len(self.ts))

        # self.assertTrue(isinstance(score, np.ndarray o list))
        # Assert comprobamos que se agregan los datos correctamente
        self.assertIsNone(model.get_x_test())

        model.set_x_test(self.ts)
        self.assertEqual(len(model.get_x_test()), len(self.ts))
        random.seed(0)
        # Assert Predicciones
        try:
            predict = model.predict()
        except Exception as e:
            raise Exception("Error al generar las predicciones")

        # self.assertTrue(isinstance(score, np.ndarray o list))
        self.assertEqual(len(predict), len(self.ts))

        # Assert Savemodel
        path_model = "model.joblib"
        model.save_model(path_model)
        self.assertTrue(os.path.exists(path_model))
        # Assert Loadmodel

        model.load_model(path_model)
        random.seed(0)
        # Assert Predicciones
        try:
            predict_load_model = model.predict()
        except Exception as e:
            raise Exception("Error al generar las predicciones")

        # self.assertTrue(isinstance(score, np.ndarray o list))
        self.assertEqual(len(predict_load_model), len(self.ts))
        # self.assertEqual(predict_load_model, predict)
        # os.remove(path_model)

    def check_without_default_model(self, model, function_model):
        """
            Performs unit tests without default values

            Args:
                - model (str): Instance of a model

            Returns:
                function_model (str): Instance of a class
        """

        hyper_model = model.get_hyperparameter()
        hyper_default = utils.ParameterManagement(function_model.__init__)._default_parameters

        # Assert Entrenar
        try:
            score = model.fit()
        except Exception as e:
            raise Exception("Error al entrenar")

        # self.assertEqual(len(score), len(self.ts))

        # self.assertTrue(isinstance(score, np.ndarray o list))
        # Assert comprobamos que se agregan los datos correctamente
        # self.assertIsNone(model.get_x_test())

        # self.assertEqual(len(model.get_x_test()), len(self.ts))
        random.seed(0)
        # Assert Predicciones
        # try:
        #     predict = model.predict()
        # except Exception as e:
        #    raise Exception("Error al generar las predicciones")

        # self.assertTrue(isinstance(score, np.ndarray o list))
        # self.assertEqual(len(predict), len(self.ts))

        # Assert Savemodel
        path_model = "model.joblib"
        model.save_model(path_model)
        self.assertTrue(os.path.exists(path_model))
        # Assert Loadmodel

        model.load_model(path_model)
        random.seed(0)
        # Assert Predicciones
        try:
            predict_load_model = model.predict()
        except Exception as e:
            raise Exception("Error al generar las predicciones")

        # self.assertTrue(isinstance(score, np.ndarray o list))
        # self.assertEqual(len(predict_load_model), len(self.ts))
        # self.assertEqual(predict_load_model, predict)
        # os.remove(path_model)


    def check_default_model_sklearn(self, model, function_model):
        """
            Performs unit tests with default values for those algorithms corresponding to sklearn.

            Args:
                - model (str): Instance of a model

            Returns:
                function_model (str): Instance of a class
        """
        hyper_model = model.get_hyperparameter()
        hyper_default = utils.ParameterManagement(function_model.__init__)._default_parameters

        # Assert de comprobar que son iguales
        self.assertEqual(hyper_model, hyper_default)
        # Assert Entrenar
        try:
            score = model.fit()
        except Exception as e:
            raise Exception("Error al entrenar")

        self.assertEqual(len(score), len(model.get_x_train()))

        # self.assertTrue(isinstance(score, np.ndarray o list))
        # Assert comprobamos que se agregan los datos correctamente
        random.seed(0)
        # Assert Predicciones
        try:
            predict = model.predict()
        except Exception as e:
            raise Exception("Error al generar las predicciones")

        # self.assertTrue(isinstance(score, np.ndarray o list))
        self.assertEqual(len(predict), len(model.get_y_test()))

        # Assert Savemodel
        path_model = "model.joblib"
        model.save_model(path_model)
        self.assertTrue(os.path.exists(path_model))
        # Assert Loadmodel

        model.load_model(path_model)
        random.seed(0)
        # Assert Predicciones
        try:
            predict_load_model = model.predict()
        except Exception as e:
            raise Exception("Error al generar las predicciones")

        # self.assertTrue(isinstance(score, np.ndarray o list))
        self.assertEqual(len(predict_load_model), len(model.get_y_test()))
        # self.assertEqual(predict_load_model, predict)
        # os.remove(path_model)

    # --------------------- Test Machine Learning Module  ---------------------

    # TODO REVISAR ESTA FUNCIÓN, INTRODUCIR OTRO CONJUNTO DE DATOS
    def test_decisionTreeClass(self):
        """
            Tests that the decision tree classifier model correctly performs the unitests functions.
        """
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=2, contamination=0.05)
        model = machine_learning.decision_tree_classifier.DecisionTreeClassifierClassOD({},X_train, y_train, X_test, y_test)
        self.check_default_model_sklearn(model, tree.DecisionTreeClassifier)

    def test_ecod(self):
        """
            Tests that the Unsupervised Outlier Detection Using
            Empirical Cumulative Distribution Functions (ECOD) model correctly performs the unitests functions.
        """
        model = machine_learning.ecod.ECOD({}, self.ts)
        self.check_default_model(model, ecod.ECOD)

    def test_iforest(self):
        """
            Tests that the IsolationForest Outlier Detector model correctly performs the unitests functions.
        """
        model = machine_learning.iforest.IForest({}, self.ts)
        self.check_default_model(model, iforest.IForest)

    def test_knn(self):
        """
            Tests that the k-Nearest Neighbors Detector (kNN) model correctly performs the unitests functions.
        """
        model = machine_learning.knn.KNN({}, self.ts)
        self.check_default_model(model, knn.KNN)


    def test_lof(self):
        """
            Tests that the Local Outlier Factor (LOF) model correctly performs the unitests functions.
        """
        model = machine_learning.lof.LOF({}, self.ts)
        self.check_default_model(model, lof.LOF)

    def test_mlp_class(self):
        """
            Tests that the Multi-layer Perceptron (MLP) model correctly performs the unitests functions.
        """
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=2, contamination=0.05)
        model = machine_learning.mlp_classifier.MLPClassOD({}, X_train, y_train, X_test, y_test)
        self.check_default_model_sklearn(model, neural_network.MLPClassifier)

    # TODO Falla al entrenar
    def test_multiNomialNB(self):
        """
            Tests that the multinomial Naive Bayes (multiNomialNB) model correctly performs the unitests functions.
        """
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=2, contamination=0.05)
        
        # Multinomial NB does not accept negative values in input data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        model = machine_learning.multinomial_nb.MultinomialNBClassOD({}, X_train, y_train, X_test, y_test)
        self.check_default_model_sklearn(model, naive_bayes.MultinomialNB)

    def test_Naive_Bayes(self):
        """
            Tests that the Naive Bayes classifier model correctly performs the unitests functions.
        """
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=2, contamination=0.05)
        model = machine_learning.naive_bayes.GaussianNBClassOD({}, X_train, y_train, X_test, y_test)
        self.check_default_model_sklearn(model, naive_bayes.GaussianNB)

    def test_ocsvm(self):
        """
            Tests that the One-class SVM detector (OSCVM) correctly performs the unitests functions.
        """
        model = machine_learning.ocsvm.OCSVM({}, self.ts)
        self.check_default_model(model, ocsvm.OCSVM)

    def test_randomForestClassifier(self):
        """
            Tests that the random forest classifier model correctly performs the unitests functions.
        """
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=2, contamination=0.05)
        model = machine_learning.random_forest_classifier.RandomForestClassOD({}, X_train, y_train, X_test, y_test)
        self.check_default_model_sklearn(model, ensemble.RandomForestClassifier)

    def test_svm(self):
        """
            Tests that the SVM model correctly performs the unitests functions.
        """
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=2, contamination=0.05)
        model = machine_learning.svm.SVMClassOD({}, X_train, y_train, X_test, y_test)
        self.check_default_model_sklearn(model, svm.SVC)
    # --------------------- Test Deep Learning Module  ---------------------

    # TODO (Falla al cargar el modelo) -> Fallan por ahora solo en Window
    def test_alad(self):
        """
            Tests that the Adversarially Learned Anomaly Detection (ALAD) model correctly performs the unitests functions.
        """
        model = deep_learning.alad.ALAD({}, self.ts)
        self.check_default_model(model, alad.ALAD)

    # TODO (Falla al cargar el modelo) -> Fallan por ahora solo en Window
    def test_anogan(self):
        """
            Tests that the Anomaly Detection with Generative Adversarial Networks  (AnoGAN) model correctly performs the unitests functions.
        """
        model = deep_learning.anogan.AnoGAN({}, self.ts)
        self.check_default_model(model, anogan.AnoGAN)

    def test_auto_encoder(self):
        """
            Tests that the auto encoder model correctly performs the unitests functions.
        """
        X_train, X_test, _, _ = generate_data(n_train=190, n_test=10, n_features=2, contamination=0.05)
 
        model = deep_learning.auto_encoder.AutoEncoder({"hidden_neurons":[2,1,2]}, X_train, None, X_test, None)
        self.check_without_default_model(model, auto_encoder.AutoEncoder)

    def test_deep_svdd(self):
        """
            Tests that the Deep One-Class Classification (SVDD) model correctly performs the unitests functions.
        """
        model = deep_learning.deep_svdd.DeepSVDD({}, self.ts)
        self.check_default_model(model, deep_svdd.DeepSVDD)

    # TODO (Falla al cargar el modelo) -> Fallan por ahora solo en Window
    def test_mo_gaal(self):
        """
            Tests that the Multiple-Objective Generative Adversarial Active Learning model correctly performs the unitests functions.
        """
        model = deep_learning.mo_gaal.MO_GAAL({}, self.ts)
        self.check_default_model(model, mo_gaal.MO_GAAL)

    # TODO (Falla al cargar el modelo) -> Fallan por ahora solo en Window
    def test_so_gaal(self):
        """
            Tests that the Single-Objective Generative Adversarial Active Learning model correctly performs the unitests functions.
        """
        model = deep_learning.so_gaal.SO_GAAL({}, self.ts)
        self.check_default_model(model, so_gaal.SO_GAAL)

    # TODO (Falla al entrenar el modelo: Arreglado) -> Falla al guadarse y cargarse
    def test_vae(self):
        """
            Tests that the Variational Auto Encoder (VAE) model correctly performs the unitests functions.
        """
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=2, contamination=0.05)
        hyperparameter = {"encoder_neurons": [2, 1], "decoder_neurons": [1, 2]}
        model = deep_learning.vae.VAE(hyperparameter, X_train, y_train, X_test, y_test)
        self.check_without_default_model(model, vae.VAE)

    # --------------------- Test Statistical Module  ---------------------
    def test_lunar(self):
        """
            Tests that the lunar model correctly performs the unitests functions.
        """
        model = statistical.lunar.LUNAR({}, self.ts)
        self.check_default_model(model, lunar.LUNAR)

    def test_abod(self):
        """
            Tests that the Angle-based Outlier Detector (ABOD) model correctly performs the unitests functions.
        """
        model = statistical.abod.ABOD({}, self.ts)
        self.check_default_model(model, abod.ABOD)

    def test_cblof(self):
        """
            Tests that the Clustering Based Local Outlier Factor (CBLOF) model correctly performs the unitests functions.
        """
        model = statistical.cblof.CBLOF({}, self.ts)
        self.check_default_model(model, cblof.CBLOF)

    # TODO (Error al entrenar -> Arreglado pero falla el tema de que no permite series temporales unidimensionales)
    def test_cd(self):
        """
            Tests that the Cook's distance outlier detection (CD) model correctly performs the unitests functions.
        """
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=2, contamination=0.05)
        # hyperparameter = {"encoder_neurons": [2, 1], "decoder_neurons": [1, 2]}
        model = statistical.cd.CD({}, X_train, y_train, X_test, y_test)
        self.check_without_default_model(model, cd.CD)

    def test_copod(self):
        """
            Tests that the Copula Based Outlier Detector (COPOD) model correctly performs the unitests functions.
        """
        model = statistical.copod.COPOD({}, self.ts)
        self.check_default_model(model, copod.COPOD)

    # TODO (Error al entrenar -> Arreglado pero falla el tema de que no permite series temporales unidimensionales)
    def test_feature_bagging(self):
        """
            Tests that the Feature bagging detector model correctly performs the unitests functions.
        """
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=2, contamination=0.05)
        hyperparameter = {"encoder_neurons": [2, 1], "decoder_neurons": [1, 2]}
        model = statistical.feature_bagging.FeatureBagging({}, X_train, y_train, X_test, y_test)
        self.check_without_default_model(model, feature_bagging.FeatureBagging)

    def test_gmm(self):
        """
            Tests that the Gaussian Mixture Model (GMM) correctly performs the unitests functions.
        """
        model = statistical.gmm.GMM({}, self.ts)
        self.check_default_model(model, gmm.GMM)

    def test_hbos(self):
        """
            Tests that the Histogram-based Outlier Detection (HBOS) model correctly performs the unitests functions.
        """
        model = statistical.hbos.HBOS({}, self.ts)
        self.check_default_model(model, hbos.HBOS)

    def test_inne(self):
        """
            Tests that the Isolation-based anomaly detection using nearest-neighbor ensembles model correctly performs the unitests functions.
        """
        model = statistical.inne.INNE({}, self.ts)
        self.check_default_model(model, inne.INNE)

    def test_kde(self):
        """
            Tests that the Kernel Density Estimation (KDE) model correctly performs the unitests functions.
        """
        model = statistical.kde.KDE({}, self.ts)
        self.check_default_model(model, kde.KDE)

    def test_kpca(self):
        """
            Tests that the Kernel Principal Component Analysis (KPCA) model correctly performs the unitests functions.
        """
        model = statistical.kpca.KPCA({}, self.ts)
        self.check_default_model(model, kpca.KPCA)

    def test_lmdd(self):
        """
            Tests that the Linear Model Deviation-base outlier detection (LMDD) model correctly performs the unitests functions.
        """
        model = statistical.lmdd.LMDD({}, self.ts)
        self.check_default_model(model, lmdd.LMDD)

    def test_loci(self):
        """
            Tests that the Local Correlation Integral model correctly performs the unitests functions.
        """
        model = statistical.loci.LOCI({}, self.ts)
        self.check_default_model(model, loci.LOCI)

    def test_loda(self):
        """
            Tests that the Lightweight on-line detector of anomalies model correctly performs the unitests functions.
        """
        model = statistical.loda.LODA({}, self.ts)
        self.check_default_model(model, loda.LODA)

    # TODO (Fallo al Crear modelo)
    def test_lscp(self):
        """
            Tests that the Locally Selective Combination of Parallel Outlier Ensembles model correctly performs the unitests functions.
        """
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=2, contamination=0.05)
        model = statistical.lscp.LSCP({"detector_list":[lof.LOF(), lof.LOF()]}, X_train,y_train, X_test,y_test)
        self.check_without_default_model(model, lscp.LSCP)

    def test_mad(self):
        """
            Tests that the Median Absolute deviation model correctly performs the unitests functions.
        """
        model = statistical.mad.MAD({}, self.ts)
        self.check_default_model(model, mad.MAD)

    def test_mcd(self):
        """
            Tests that the Outlier Detection with Minimum Covariance Determinant (MCD) model correctly performs the unitests functions.
        """
        model = statistical.mcd.MCD({}, self.ts)
        self.check_default_model(model, mcd.MCD)

    def test_pca(self):
        """
            Tests that the Principal Component Analysis (PCA) model correctly performs the unitests functions.
        """
        model = statistical.pca.PCA({}, self.ts)
        self.check_default_model(model, pca.PCA)

    def test_qmcd(self):
        """
            Tests that the Quasi-Monte Carlo Discrepancy outlier detection (QMCD) model correctly performs the unitests functions.
        """
        model = statistical.qmcd.QMCD({}, self.ts)
        self.check_default_model(model, qmcd.QMCD)

    def test_rgraph(self):
        """
            Tests that the R-graph model correctly performs the unitests functions.
        """
        model = statistical.rgraph.RGraph({}, self.ts)
        self.check_default_model(model, rgraph.RGraph)

    def test_rod(self):
        """
            Tests that the Rotation-based Outlier Detector (ROD) model correctly performs the unitests functions.
        """
        model = statistical.rod.ROD({}, self.ts)
        self.check_default_model(model, rod.ROD)

    def test_sampling(self):
        """
            Tests that the Outlier detection based on Sampling (SP) model correctly performs the unitests functions.
        """
        model = statistical.sampling.Sampling({}, self.ts)
        self.check_default_model(model, sampling.Sampling)

    def test_sod(self):
        """
            Tests that the Subspace Outlier Detection (SOD) model correctly performs the unitests functions.
        """
        model = statistical.sod.SOD({}, self.ts)
        self.check_default_model(model, sod.SOD)

    def test_sos(self):
        """
            Tests that the Stochastic Outlier Selection (SOS) model correctly performs the unitests functions.
        """
        model = statistical.sos.SOS({}, self.ts)
        self.check_default_model(model, sos.SOS)



    # TODO (Falla Falta el conjunto de datos (que tenga X e y))
    def test_xgbod(self):
        """
            Tests that the Improving Supervised Outlier Detection with Unsupervised (SGBOD) model correctly performs the unitests functions.
        """
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=2, contamination=0.05)
        hyperparameter = {"encoder_neurons": [2, 1], "decoder_neurons": [1, 2]}
        model = statistical.xgbod.XGBOD({}, X_train, y_train, X_test, y_test)
        self.check_without_default_model(model, xgbod.XGBOD)

    def test_mp(self):
        """
            Tests that the Matrix Profile (MP) model correctly performs the unitests functions.
        """
        x_train, x_test, y_train, y_test = generate_data(n_train=1000, n_test=1000, n_features=2, contamination=0.05)
        hyperparameter = {"m": 10, "threshold": 0.5}
        x_train = x_train.transpose()
        x_test = x_test.transpose()
        model = statistical.matrixprofile.MatrixProfile(hyperparameter, x_train, y_train, x_test, y_test)

        hyper_model = model.get_hyperparameter()

        function_model, parameter_t = (stumpy.mstump, "T") if x_train.shape[0] > 1 else (stumpy.stump, "T_A")
        hyper_default = utils.ParameterManagement(function_model)._default_parameters
        hyper_default["m"] = hyper_model["m"]
        hyper_default[parameter_t] = hyper_model[parameter_t]
        hyper_default["threshold"] = hyper_model["threshold"]
        # Assert de comprobar que son iguales
        self.assertEqual(hyper_model, hyper_default)
        # Assert Entrenar
        try:
            score = model.fit()
        except Exception as e:
            raise Exception("Error al entrenar")

        # self.assertTrue(isinstance(score, np.ndarray o list))
        # Assert comprobamos que se agregan los datos correctamente
        # self.assertIsNone(model.get_x_test())

        self.assertEqual(model.get_x_test().shape, x_test.shape)
        random.seed(0)
        # Assert Predicciones
        try:
            predict = model.predict()
        except Exception as e:
            raise Exception("Error al generar las predicciones")
        

    def test_arima(self):
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=1, contamination=0.05)
        model = statistical.arima.ARIMA({"endog":X_train.flatten()}, None, None, X_test, None)
        self.check_without_default_model(model, arima.ARIMA)
    
    def test_sarimax(self):
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=1, contamination=0.05)
        model = statistical.sarimax.SARIMAX({"endog":X_train}, None, None, X_test, None)
        self.check_without_default_model(model, sarimax.SARIMAX)
    
    def test_varmax(self):
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=2, contamination=0.05)
        model = statistical.varmax.VARMAX({"endog": X_train}, None, None, X_test, None)
        self.check_without_default_model(model, varmax.VARMAX)

    def test_holtwinters(self):
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=1, contamination=0.05)
        model = statistical.holt_winters.HoltWinters({"endog": X_train}, None, None, X_test, None)
        self.check_without_default_model(model, holtwinters.Holt)

    def test_singleexponentialsmoothing(self):
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=10, n_features=1, contamination=0.05)
        model = statistical.singleExponentialSmoothing.SingleExponentialSmoothing({"endog":X_train.flatten()}, None, None, X_test, None)
        self.check_without_default_model(model, ets.ETSModel)


    @staticmethod
    def format_time_series(X, Y, time_steps, stride):
        x = []
        y = []

        for i in range(0, len(Y) -time_steps, stride):
            x.append(X[i:i + time_steps])
            y.append(Y[i + time_steps])

        x = np.array(x)
        y = np.array(y)

        return x, y

    def test_supervised_topology(self):

        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=100, n_features=3, contamination=0.05)
        X_train, y_train = self.format_time_series(X_train, y_train, 5, 1)
        X_test, y_test = self.format_time_series(X_test, y_test, 5, 1)

        layer0 = deep_learning.layers.Layer(deep_learning.layers.LayerType.CNN, user_hyperparameters={"filters":32,  "kernel_size":5}).layer
        layer2 = deep_learning.layers.Layer(deep_learning.layers.LayerType.LSTM, user_hyperparameters={"units": 100}).layer
        
        hyperparam = {"input_shape": (5, 3),
                    "layers": [layer0, layer2],
                    "compile": {"optimizer":"adam", "loss":"mse"},
                    "fit": {"x": X_train, "y": y_train}
                }
        
        model = deep_learning.topology.Topology(hyperparam)
        model.fit()

        print(model.predict(X_test))


    def test_copy_layers_topology(self):
        X_train, X_test, y_train, y_test = generate_data(n_train=190, n_test=100, n_features=3, contamination=0.05)
        X_train, y_train = self.format_time_series(X_train, y_train, 5, 1)
        X_test, y_test = self.format_time_series(X_test, y_test, 5, 1)

        layer0 = deep_learning.layers.Layer(deep_learning.layers.LayerType.CNN, user_hyperparameters={"filters":3,  "kernel_size":2}).layer
        layer2 = deep_learning.layers.Layer(deep_learning.layers.LayerType.LSTM, user_hyperparameters={"units": 100}).layer
        
        hyperparam = {"input_shape": (5, 3),
                    "layers": [layer0, layer0, layer2],
                    "compile": {"optimizer":"adam", "loss":"mse"},
                    "fit": {"x": X_train, "y": X_train}
                }

        model = deep_learning.topology.Topology(hyperparam)
        model.fit({"x": X_train, "y": y_train})

        print(model.predict(X_test))

    def test_autoencoder_copy_layers_cnn(self):

        X_train, X_test, y_train, y_test = generate_data(n_train=201, n_test=201, n_features=3, contamination=0.05)
        X_train, y_train = self.format_time_series(X_train, y_train, 100, 100)
        X_test, y_test = self.format_time_series(X_test, y_test, 100, 100)
  
        print(X_train)

        layerCNN16 = deep_learning.layers.Layer(deep_learning.layers.LayerType.CNN, user_hyperparameters={"filters":16,  "kernel_size":5, "padding":'same'}).layer
        
        layerMAXPooling = deep_learning.layers.Layer(deep_learning.layers.LayerType.MAXPOOLING_1D, user_hyperparameters={"pool_size":2,  "padding":'same'}).layer
        layerCNN8 = deep_learning.layers.Layer(deep_learning.layers.LayerType.CNN, user_hyperparameters={"filters":8,  "kernel_size":5, "padding":'same'}).layer
        layerUPSAMPLING = deep_learning.layers.Layer(deep_learning.layers.LayerType.UPSAMPLING_1D, user_hyperparameters={"size":2}).layer

        layerOUT = deep_learning.layers.Layer(deep_learning.layers.LayerType.CNN, user_hyperparameters={"filters":3,  "kernel_size":5, "padding":'same'}).layer


        autoencoder_arc = [layerCNN16,layerMAXPooling,layerCNN8,layerMAXPooling,layerCNN8,layerUPSAMPLING,layerCNN16,layerUPSAMPLING,layerOUT ]

        hyperparam = {"input_shape": (100, 3),
                    "layers": autoencoder_arc,
                    "compile": {"optimizer":"adam", "loss":"mse"},
                    "fit": {"x": X_train, "y": X_train}
                }

        model = deep_learning.topology.Topology(hyperparameter=hyperparam)

        model.fit()

        pred = model.predict(X_test)

  


    def test_autoencoder_cnn(self):

        X_train, X_test, y_train, y_test = generate_data(n_train=201, n_test=201, n_features=3, contamination=0.05)
        X_train, y_train = self.format_time_series(X_train, y_train, 100, 100)
        X_test, y_test = self.format_time_series(X_test, y_test, 100, 100)
  
        print(X_train)

        layer0 = deep_learning.layers.Layer(deep_learning.layers.LayerType.CNN, user_hyperparameters={"filters":16,  "kernel_size":5, "padding":'same'}).layer
        layer1 = deep_learning.layers.Layer(deep_learning.layers.LayerType.MAXPOOLING_1D, user_hyperparameters={"pool_size":2,  "padding":'same'}).layer
        layer2 = deep_learning.layers.Layer(deep_learning.layers.LayerType.CNN, user_hyperparameters={"filters":8,  "kernel_size":5, "padding":'same'}).layer
        layer3 = deep_learning.layers.Layer(deep_learning.layers.LayerType.MAXPOOLING_1D, user_hyperparameters={"pool_size":2,  "padding":'same'}).layer
        layer4 = deep_learning.layers.Layer(deep_learning.layers.LayerType.CNN, user_hyperparameters={"filters":8,  "kernel_size":5, "padding":'same'}).layer
        layer5 = deep_learning.layers.Layer(deep_learning.layers.LayerType.UPSAMPLING_1D, user_hyperparameters={"size":2}).layer
        layer6 = deep_learning.layers.Layer(deep_learning.layers.LayerType.CNN, user_hyperparameters={"filters":16,  "kernel_size":5, "padding":'same'}).layer
        layer7 = deep_learning.layers.Layer(deep_learning.layers.LayerType.UPSAMPLING_1D, user_hyperparameters={"size":2}).layer
        layer8 = deep_learning.layers.Layer(deep_learning.layers.LayerType.CNN, user_hyperparameters={"filters":3,  "kernel_size":5, "padding":'same'}).layer



        autoencoder_arc = [layer0,layer1,layer2,layer3,layer4,layer5,layer6,layer7,layer8 ]

        hyperparam = {"input_shape": (100, 3),
                        "layers": autoencoder_arc,
                        "compile": {"optimizer":"adam", "loss":"mse"},
                        "fit": {"x": X_train, "y": X_train}
                    }

        model = deep_learning.topology.Topology(hyperparameter=hyperparam)

        model.fit()

        pred = model.predict_reconstruction_error(X_test, X_test)

        print(pred)

