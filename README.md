# Hi, anomaly_lib is a library for statistical testing and comparison of algorithm results 👋

![GitHub Followers](https://img.shields.io/github/followers/kdis-lab?style=social)

## Anomaly Library (X)

A time series anomaly detection library featuring cutting-edge algorithms and advanced functionalities.It allows the unification of more than 50 algorithms from the literature distributed across different libraries, in addition to some of its own.

In detail, the package provides:



### Available algorithms:

#### **Deep Learning**

| Name      | Function                                             |
|-----------|------------------------------------------------------|
| ALAD      | anomaly_lib.deep_learning.alad.ALAD                     |
| AnoGAN    | anomaly_lib.deep_learning.anogan.AnoGAN                 |
| AutoEncoder | anomaly_lib.deep_learning.auto_encoder.AutoEncoder      |
| DeepSVDD  | anomaly_lib.deep_learning.deep_svdd.DeepSVDD            |
| MO_GAAL   | anomaly_lib.deep_learning.mo_gaal.MO_GAAL               |
| SO_GAAL   | anomaly_lib.deep_learning.so_gaal.SO_GAAL               |
| VAE       | anomaly_lib.deep_learning.vae.VAE                       |


#### **Machine Learning**

| Name                          | Function                                                                         |
|-------------------------------|----------------------------------------------------------------------------------|
| DecisionTreeClassifierClassOD | anomaly_lib.machine_learning.decision_tree_classifier.DecisionTreeClassifierClassOD |
| ECOD                          | anomaly_lib.machine_learning.ecod.ECOD                                               |
| IForest                       | anomaly_lib.machine_learning.iforest.IForest                                         |
| KNN                           | anomaly_lib.machine_learning.knn.KNN                                                 |
| LOF                           | anomaly_lib.machine_learning.lof.LOF                                                 |
| MLPClassOD                    | anomaly_lib.machine_learning.mlp_classifier.MLPClassOD                               |
| MultinomialNBClassOD          | anomaly_lib.machine_learning.multinomial_nb.MultinomialNBClassOD                     |
| GaussianNBClassOD             | anomaly_lib.machine_learning.naive_bayes.GaussianNBClassOD                           |
| OCSVM                         | anomaly_lib.machine_learning.ocsvm.OCSVM                                             |
| RandomForestClassOD           | anomaly_lib.machine_learning.random_forest_classifier.RandomForestClassOD             |
| SVMClassOD                    | anomaly_lib.machine_learning.svm.SVMClassOD                                          |

#### **Statistical**

| Name                          | Function                                                                         |
|-------------------------------|----------------------------------------------------------------------------------|
| ABOD                          | anomaly_lib.statistical.abod.ABOD                                                    |
| CBLOF                         | anomaly_lib.statistical.cblof.CBLOF                                                  |
| CD                            | anomaly_lib.statistical.cd.CD                                                        |
| COF                           | anomaly_lib.statistical.cof.COF                                                      |
| LUNAR                         | anomaly_lib.statistical.lunar.LUNAR                                                  |
| COPOD                         | anomaly_lib.statistical.copod.COPOD                                                  |
| FeatureBagging                | anomaly_lib.statistical.feature_bagging.FeatureBagging                               |
| GMM                           | anomaly_lib.statistical.gmm.GMM                                                      |
| HBOS                          | anomaly_lib.statistical.hbos.HBOS                                                    |
| INNE                          | anomaly_lib.statistical.inne.INNE                                                    |
| KDE                           | anomaly_lib.statistical.kde.KDE                                                      |
| KPCA                          | anomaly_lib.statistical.kpca.KPCA                                                    |
| LMDD                          | anomaly_lib.statistical.lmdd.LMDD                                                    |
| LOCI                          | anomaly_lib.statistical.loci.LOCI                                                    |
| LODA                          | anomaly_lib.statistical.loda.LODA                                                    |
| LSCP                          | anomaly_lib.statistical.lscp.LSCP                                                    |
| MAD                           | anomaly_lib.statistical.mad.MAD                                                      |
| MCD                           | anomaly_lib.statistical.mcd.MCD                                                      |
| PCA                           | anomaly_lib.statistical.pca.PCA                                                      |
| QMCD                          | anomaly_lib.statistical.qmcd.QMCD                                                    |
| RGraph                        | anomaly_lib.statistical.rgraph.RGraph                                                |
| ROD                           | anomaly_lib.statistical.rod.ROD                                                      |
| Sampling                      | anomaly_lib.statistical.sampling.Sampling                                            |
| SOD                           | anomaly_lib.statistical.sod.SOD                                                      |
| SOS                           | anomaly_lib.statistical.sos.SOS                                                      |
| XGBOD                         | anomaly_lib.statistical.xgbod.XGBOD                                                  |
| MatrixProfile                 | anomaly_lib.statistical.matrixprofile.MatrixProfile                                  |
| ARIMA                         | anomaly_lib.statistical.arima.ARIMA                                                  |
| SARIMAX                       | anomaly_lib.statistical.sarimax.SARIMAX                                              |
| VARMAX                        | anomaly_lib.statistical.varmax.VARMAX                                                |
| HoltWinters                   | anomaly_lib.statistical.holt_winters.HoltWinters                                    |
| SingleExponentialSmoothing    | anomaly_lib.statistical.singleExponentialSmoothing.SingleExponentialSmoothing        |


## Developed in:
![Python](https://img.shields.io/badge/Python-yellow?style=for-the-badge&logo=python&logoColor=white&labelColor=101010)


## Documentación
You can find all documentation in [Web Docs](https://github.com/kdis-lab/anomaly_lib).