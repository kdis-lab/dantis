
NAME_LIBRARY (SIGLAS)
=====================

SIGLAS es una librería para el desarrollo de modelos de detección de anomalías. Permite la unificación de más de 50 algoritmos de la literatura distribuidos en diferentes librerías, además de algunos propios. 

In detail, the package provide:

**Deep Learning**:
++++++++++++++++++
   - ALAD :class:`anomaly_lib.deep_learning.alad.ALAD`
   - AnoGAN :class:`anomaly_lib.deep_learning.anogan.AnoGAN`
   - AutoEncoder :class:`anomaly_lib.deep_learning.auto_encoder.AutoEncoder`
   - DeepSVDD :class:`anomaly_lib.deep_learning.deep_svdd.DeepSVDD`
   - MO_GAAL :class:`anomaly_lib.deep_learning.mo_gaal.MO_GAAL`
   - SO_GAAL :class:`anomaly_lib.deep_learning.so_gaal.SO_GAAL`
   - VAE :class:`anomaly_lib.deep_learning.vae.VAE`
   
**Machine Learning**:
+++++++++++++++++++++
   - DecisionTreeClassifierClassOD :class:`anomaly_lib.machine_learning.decision_tree_classifier.DecisionTreeClassifierClassOD`
   - ECOD :class:`anomaly_lib.machine_learning.ecod.ECOD`
   - IForest :class:`anomaly_lib.machine_learning.iforest.IForest`
   - KNN :class:`anomaly_lib.machine_learning.knn.KNN`
   - LOF :class:`anomaly_lib.machine_learning.lof.LOF`
   - MLPClassOD :class:`anomaly_lib.machine_learning.mlp_classifier.MLPClassOD`
   - MultinomialNBClassOD :class:`anomaly_lib.machine_learning.multinomial_nb.MultinomialNBClassOD`
   - GaussianNBClassOD :class:`anomaly_lib.machine_learning.naive_bayes.GaussianNBClassOD`
   - OCSVM :class:`anomaly_lib.machine_learning.ocsvm.OCSVM`
   - RandomForestClassOD :class:`anomaly_lib.machine_learning.random_forest_classifier.RandomForestClassOD`
   - SVMClassOD :class:`anomaly_lib.machine_learning.svm.SVMClassOD`

**Statistical**:
+++++++++++++++++++
   - ABOD :class:`anomaly_lib.statistical.abod.ABOD`
   - CBLOF :class:`anomaly_lib.statistical.cblof.CBLOF`
   - CD :class:`anomaly_lib.statistical.cd.CD`
   - COF :class:`anomaly_lib.statistical.cof.COF` 
   - LUNAR :class:`anomaly_lib.statistical.lunar.LUNAR`
   - COPOD :class:`anomaly_lib.statistical.copod.COPOD`
   - FeatureBagging :class:`anomaly_lib.statistical.feature_bagging.FeatureBagging`
   - GMM :class:`anomaly_lib.statistical.gmm.GMM`
   - HBOS :class:`anomaly_lib.statistical.hbos.HBOS`
   - INNE :class:`anomaly_lib.statistical.inne.INNE`
   - KDE :class:`anomaly_lib.statistical.kde.KDE`
   - KPCA :class:`anomaly_lib.statistical.kpca.KPCA`
   - LMDD :class:`anomaly_lib.statistical.lmdd.LMDD`
   - LOCI :class:`anomaly_lib.statistical.loci.LOCI`
   - LODA :class:`anomaly_lib.statistical.loda.LODA`
   - LSCP :class:`anomaly_lib.statistical.lscp.LSCP`
   - MAD :class:`anomaly_lib.statistical.mad.MAD`
   - MCD :class:`anomaly_lib.statistical.mcd.MCD`
   - PCA :class:`anomaly_lib.statistical.pca.PCA`
   - QMCD :class:`anomaly_lib.statistical.qmcd.QMCD`
   - RGraph :class:`anomaly_lib.statistical.rgraph.RGraph`
   - ROD :class:`anomaly_lib.statistical.rod.ROD`
   - Sampling :class:`anomaly_lib.statistical.sampling.Sampling`
   - SOD :class:`anomaly_lib.statistical.sod.SOD`
   - SOS :class:`anomaly_lib.statistical.sos.SOS`
   - XGBOD :class:`anomaly_lib.statistical.xgbod.XGBOD`
   - MatrixProfile :class:`anomaly_lib.statistical.matrixprofile.MatrixProfile`
   - ARIMA :class:`anomaly_lib.statistical.arima.ARIMA`
   - SARIMAX :class:`anomaly_lib.statistical.sarimax.SARIMAX`
   - VARMAX :class:`anomaly_lib.statistical.varmax.VARMAX`
   - HoltWinters :class:`anomaly_lib.statistical.holt_winters.HoltWinters`
   - SingleExponentialSmoothing :class:`anomaly_lib.statistical.singleExponentialSmoothing.SingleExponentialSmoothing`



.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Usage
   :hidden:

   usage/quickstart
   usage/notebooks

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package API
   :hidden:

   modules

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Index
   :hidden:

   genindex