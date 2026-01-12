
<p align="center">
  <img width="200" src="https://github.com/kdis-lab/dantis/blob/master/docs/source/_static/logo-dantis.png?raw=true" alt="DANTIS logo" style="border-radius: 30px;"/>
</p>

![GitHub Followers](https://img.shields.io/github/followers/kdis-lab?style=social)
[![PyPI version](https://img.shields.io/pypi/v/dantis.svg)](https://pypi.org/project/dantis/)
[![License: MIT](https://img.shields.io/pypi/l/dantis.svg)](https://github.com/kdis-lab/dantis/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000.svg)](https://github.com/psf/black)


--- 

DANTIS brings together **50+ anomaly detection algorithms**—from classical statistical methods and machine learning to deep learning—under a **unified scikit-learn–style API**. It consolidates models from diverse libraries and includes proprietary algorithms developed by our team.


## Installation
```bash
pip install dantis              # From PyPI (recommended)
```

## Quick start
```python
from dantis.machine_learning import DecisionTreeClassifier
from dantis.deep_learning import ALAD
from sklearn.metrics import accuracy_score
import pandas as pd

# Load your time series data
df = pd.read_csv("anomaly_datasets/anomaly_dataset_1.csv")

X = df.drop(columns=["is_anomaly"])
y = df["is_anomaly"]

# Split data chronologically for time series (no shuffling)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)



y_pred = decision_tree_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"DecisionTree Accuracy: {acc:.3f}")

alad_model = ALAD()
alad_model.fit(X_train, y_train)

y_pred = alad_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"ALAD Accuracy: {acc:.3f}")
```



## Key features

* **Wide algorithm coverage** – 50+ detectors across deep learning, machine learning, and statistical methods.
* **Consistent API** – all detectors follow the scikit-learn `fit/predict` paradigm and expose `decision_scores_`, `predict_proba`, …
* **Pipeline & preprocessing layer** – compose detectors with built-in scalers, transforms, and windowing utilities.
* **Comprehensive evaluation** – metrics and visualisation helpers for rapid benchmarking.
* **Extensibility** – subclass `AlgorithBase` and plug your own model into the ecosystem.

## GUI: Desktop interface for model training & evaluation

DANTIS includes a full-featured graphical interface to simplify the use of the library for users without programming experience. Inspired by tools like **Weka** and **KNIME**, this desktop application enables:

- Training and testing of anomaly detection models
- Preprocessing pipelines and parameter tuning
- Quantitative comparison of models using the [StaTDS](https://github.com/kdis-lab/StaTDS) library

<p align="center">
  <img src="https://github.com/kdis-lab/dantis/blob/master/docs/source/_static/gui-overview.png?raw=true" alt="DANTIS GUI" width="600"/>
</p>



## Developed in:
![Python](https://img.shields.io/badge/Python-yellow?style=for-the-badge&logo=python&logoColor=white&labelColor=101010)
![PyQT5](https://img.shields.io/badge/PyQt5-5.15.8-blue)

## Documentación
Documentation is currently available on [GitHub](https://github.com/kdis-lab/anomaly_lib).

📚 A full documentation site with tutorials and API reference is being developed at **[https://dantis.readthedocs.io](https://dantis.readthedocs.io)**.


## Funding & affiliations

DANTIS has been developed within the research activities of the DaSCI Institute (Andalusian Inter-University Institute in Data Science and Computational Intelligence), by members of the research groups KDISLAB (Knowledge Discovery and Intelligent Systems, University of Córdoba) and SCI2S (Soft Computing and Intelligent Information Systems, University of Granada).

This work has been supported by the following projects:
- TED2021-132702B-C22 - Mantenimiento Predictivo basado en Detección de Anomalías: Framework y Mantenimiento de Camiones de Alto Tonelaje (PREMAD-Truck)

We gratefully acknowledge this support.

<p align="center">
  <img src="https://github.com/kdis-lab/dantis/blob/master/docs/source/_static/logo-dasci?raw=true" alt="DaSCI" height="90"/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</br>
  <img src="https://github.com/kdis-lab/dantis/blob/master/docs/source/_static/logo-kdislab.png?raw=true" alt="KDISLAB" height="60"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/kdis-lab/dantis/blob/master/docs/source/_static/logo-sci2s.png?raw=true" alt="SCI2S" height="70"/>
</p>

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


## License

`DANTIS` is distributed under the **MIT License**.
See [`LICENSE`](LICENSE) for details.

## Acknowledgments
This research was supported in part by the PID2023-148396NB-I00 and TED2021-132702B-C22 projects of Spanish Ministry of Science and Innovation and the European Regional Development Fund, by the ProyExcel-0069 project of the Andalusian University, Research and Innovation Department.

## Supported Datasets

DANTIS provides unified loaders for a wide range of Time Series Anomaly Detection (TSAD) datasets, consolidating resources from major benchmarks like TSB-AD, NAB, and UCR.

| Dataset | Real/Synth | Type | Domain | License | Access |
| --- | --- | --- | --- | --- | --- |
| **[CalIt2](https://archive.ics.uci.edu/ml/datasets/CalIt2+Building+People+Counts)** | Real | MTS | Urban events | Unknown | Direct |
| **[CAP](https://physionet.org/content/capslpdb/1.0.0/)** | Real | MTS | Medical | Unknown | PhysioNet (Auth) |
| **[CATSv2](https://www.google.com/search?q=https://doi.org/10.5281/zenodo.3678238)** | Unknown | UTS | Simulated System | CC BY 4.0 | Direct |
| **[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)** | Real | MTS | Server/Network | Unknown | Direct |
| **[Credit Card](https://www.openml.org/search?type=data&sort=runs&id=1597)** | Real | MTS | Fraud Detection | None | OpenML |
| **[Daphnet](https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait)** | Real | UTS | Medical | CC BY 4.0 | Direct |
| **[DMDS](https://iair.mchtr.pw.edu.pl/Damadics)** | Real | MTS | ICS | Unknown | Direct |
| **[Dodgers Loop](https://archive.ics.uci.edu/ml/datasets/dodgers+loop+sensor)** | Real | UTS | Urban Traffic | Unknown | Direct |
| **[Engine Dataset](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)** | Real | MTS | ICS | Unknown | Direct |
| **[Exathlon](https://github.com/exathlonbenchmark/exathlon)** | Real | MTS | Server Monitoring | Apache-2.0 | GitHub |
| **[GECCO IoT](https://zenodo.org/record/3884398)** | Real | MTS | IoT | CC BY 4.0 | Direct |
| **[Genesis](https://www.kaggle.com/inIT-OWL/genesis-demonstrator-data-for-machine-learning)** | Real | MTS | ICS | CC BY-NC-SA 4.0 | Kaggle (Auth) |
| **[GHL](https://kas.pr/ics-research/dataset_ghl_1)** | Synth | MTS | ICS | None | Direct |
| **[IOPS](https://github.com/iopsai/iops)** | Real | UTS | Business | None | GitHub |
| **[Ionosphere](https://search.r-project.org/CRAN/refmans/fdm2id/html/ionosphere.html)** | Real | MTS | Astronomy | Unknown | Direct |
| **[KDDCUP99](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)** | Real | MTS | Computer Networks | Unknown | Direct |
| **[Kitsune](https://archive.ics.uci.edu/ml/datasets/Kitsune+Network+Attack+Dataset)** | Real | MTS | Computer Networks | Unknown | Direct |
| **[KPI AIOPS](https://competition.aiops-challenge.com/home/competition)** | Real | UTS | Business | Unknown | Competition |
| **[MBD](https://github.com/QAZASDEDC/TopoMAD)** | Real | MTS | Server Monitoring | Unknown | GitHub |
| **[Metro](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)** | Real | MTS | Urban Traffic | Unknown | Direct |
| **[MGAB](https://github.com/MarkusThill/MGAB/)** | Synth | UTS | Medical | CC0-1.0 | GitHub |
| **[MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/1.0.0/)** | Real | MTS | Medical/ECG | ODC-By 1.0 | PhysioNet (Auth) |
| **[MIT-BIH-LTDB](https://doi.org/10.13026/C2KS3F)** | Real | UTS | Medical/ECG | ODC-By 1.0 | PhysioNet (Auth) |
| **[MIT-BIH-SVDB](https://doi.org/10.13026/C2V30W)** | Real | MTS | Medical/ECG | ODC-By 1.0 | PhysioNet (Auth) |
| **[MMS](https://github.com/QAZASDEDC/TopoMAD)** | Real | MTS | Server Monitoring | Unknown | GitHub |
| **[MSL](https://github.com/khundman/telemanom)** | Real | MTS | Aerospace | Caltech | GitHub |
| **[NAB (subsets)](https://github.com/numenta/NAB)** | Real/Synth | UTS | Multiple | GPL | GitHub |
| **[NASA Shuttle](https://cs.fit.edu/~pkc/nasa/data/)** | Real | MTS | Aerospace | Unknown | Direct |
| **[NEK](https://www.google.com/search?q=https://github.com/mribrahim/TSA)** | Unknown | Unknown | Network | None | Verify Source |
| **[NeurIPS-TS](https://github.com/datamllab/tods/tree/benchmark/benchmark/synthetic)** | Synth | UTS | Multiple | Unknown | GitHub |
| **[NormA](https://helios2.mi.parisdescartes.fr/~themisp/norma/)** | Real/Synth | UTS | Multiple | Unknown | Direct |
| **[NYC Bike](https://ride.citibikenyc.com/system-data)** | Real | Both | Urban Transport | Unknown | Direct |
| **[NYC Taxi](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)** | Real | Both | Urban Transport | Unknown | Direct |
| **[OPPORTUNITY](https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition)** | Real | MTS | Activity Recog. | CC BY 4.0 | Direct |
| **[Power Demand](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)** | Real | UTS | ICS | None | Direct |
| **[PSM](https://github.com/eBay/RANSynCoders)** | Real | MTS | Server Metrics | CC BY 4.0 | GitHub |
| **[PUMP](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)** | Real | MTS | ICS | Unknown | Kaggle (Auth) |
| **[SED](https://data.nasa.gov/)** | Unknown | Unknown | Industrial | None | Verify Source |
| **[SMAP](https://github.com/khundman/telemanom)** | Real | MTS | Environmental | Caltech | GitHub |
| **[SMD](https://github.com/NetManAIOps/OmniAnomaly/)** | Real | MTS | Server Monitoring | MIT | GitHub |
| **[SWAN-SF](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM)** | Real | MTS | Astronomy | Unknown | Direct |
| **[SWaT](http://itrust.sutd.edu.sg/research/testbeds/secure-water-treatment-swat/)** | Real | MTS | ICS/Water | Request | Request Form |
| **[SensoreScope](https://doi.org/10.5281/zenodo.2654726)** | Real | UTS | IoT | Unknown | Direct |
| **[Space Shuttle](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)** | Real | UTS | Aerospace | Unknown | Direct |
| **[Stock](https://www.google.com/search?q=https://github.com/mribrahim/TSA)** | Unknown | Unknown | Finance | None | Verify Source |
| **[TODS](https://github.com/datamllab/tods)** | Synth | Unknown | Multiple | Apache-2.0 | GitHub |
| **[UCR](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)** | Real/Synth | Both | Multiple | None | Direct |
| **[WADI](http://itrust.sutd.edu.sg/testbeds/water-distribution-wadi/)** | Real | MTS | ICS/Water | Unknown | Request Form |
| **[WSD](https://www.google.com/search?q=https://github.com/mribrahim/TSA)** | Unknown | Unknown | Web Services | None | Verify Source |
| **[WaterLog](https://www.google.com/search?q=https://github.com/mribrahim/TSA)** | Real | MTS | ICS | Unknown | Direct |
| **[Yahoo](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70)** | Real/Synth | UTS | Multiple | Yahoo Terms | Request (Auth) |

> **Note:** **MTS** = Multivariate Time Series, **UTS** = Univariate Time Series. Access types marked as "Auth" or "Request" may require creating an account or filling out a form on the provider's website. DANTIS provides helper functions to facilitate the loading of these datasets once acquired.
