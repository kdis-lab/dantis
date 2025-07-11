from PyQt5.QtWidgets import QMainWindow
from view.sidebar_ui import Ui_MainWindow

from view.datasets_view import ViewDatasets
from view.models_view import ViewModels
from view.metrics_view import ViewMetrics
from view.validation_view import ViewValidation
from view.results_view import ViewResults

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Carga la interfaz y se palica la interfaz a al ventana
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Crear vistas
        self.view_datasets = ViewDatasets(self.ui)
        self.ui.stackedWidget.addWidget(self.view_datasets)

        self.view_validation = ViewValidation(self, self.ui, self.view_datasets.datasetController)
        self.ui.stackedWidget.addWidget(self.view_validation)

        self.view_models = ViewModels(self, self.ui, 
                                      self.view_datasets.datasetController, 
                                      self.view_validation.validationController)
        self.ui.stackedWidget.addWidget(self.view_models)

        self.view_metrics = ViewMetrics(self, self.ui)
        self.ui.stackedWidget.addWidget(self.view_metrics)

        self.view_results = ViewResults(self, self.ui, 
                                        self.view_datasets.datasetController,
                                        self.view_models.modelController, 
                                        self.view_metrics.metricsController, 
                                        self.view_validation.validationController )
        self.ui.stackedWidget.addWidget(self.view_results)

        self.vistas = [
            self.view_datasets,      # índice 0
            self.view_validation,    # índice 1
            self.view_models,        # índice 2
            self.view_metrics,       # índice 3
            self.view_results        # índice 4
        ]

        self.ui.icon_only_widget.hide()
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.datasets_btn_1.setChecked(True)
        self.ui.datasets_btn_2.setChecked(True)
        self.conectar_botones()


    def conectar_botones(self):
        botones_por_indice = {
            0: [self.ui.datasets_btn_1, self.ui.datasets_btn_2],
            1: [self.ui.val_btn_1, self.ui.val_btn_2],
            2: [self.ui.model_btn_1, self.ui.model_btn_2],
            3: [self.ui.metrics_btn_1, self.ui.metrics_btn_2],
            4: [self.ui.results_btn_1, self.ui.results_btn_2],
        }

        for indice, botones in botones_por_indice.items():
            for btn in botones:
                btn.toggled.connect(lambda checked, i=indice: self.cambiar_vista_si_valida(i, checked))

    def cambiar_vista_si_valida(self, index_destino, checked):
        if not checked:
            return

        indice_actual = self.ui.stackedWidget.currentIndex()
        if indice_actual == index_destino:
            return

        vista_actual = self.vistas[indice_actual]
        if hasattr(vista_actual, 'validar_datos'):
            if not vista_actual.validar_datos():
                self.sincronizar_botones(indice_actual)
                return

        self.ui.stackedWidget.setCurrentIndex(index_destino)
        self.sincronizar_botones(index_destino)


    def sincronizar_botones(self, index):
        botones = {
            0: [self.ui.datasets_btn_1, self.ui.datasets_btn_2],
            1: [self.ui.val_btn_1, self.ui.val_btn_2],
            2: [self.ui.model_btn_1, self.ui.model_btn_2],
            3: [self.ui.metrics_btn_1, self.ui.metrics_btn_2],
            4: [self.ui.results_btn_1, self.ui.results_btn_2],
        }

        # Desmarcar todos sin emitir señales
        for grupo in botones.values():
            for btn in grupo:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)

        # Activar solo los del índice seleccionado
        for btn in botones[index]:
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)

