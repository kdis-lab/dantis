from PyQt5.QtWidgets import QMainWindow

from view.sidebar_ui import Ui_MainWindow
from view.datasets_view import ViewDatasets
from view.models_view import ViewModels
from view.metrics_view import ViewMetrics
from view.validation_view import ViewValidation
from view.results_view import ViewResults

class MainWindow(QMainWindow):
    """
    MainWindow class managing the main application window and navigation between different views.

    This class initializes the UI and creates multiple views representing different parts of the application,
    such as datasets, validation, models, metrics, and results. It also manages the navigation buttons and
    ensures that view switching occurs only when data validation passes.

    Attributes
    ----------
    ui : Ui_MainWindow
        The generated UI object containing the main window elements.
    view_datasets : ViewDatasets
        View for managing datasets.
    view_validation : ViewValidation
        View for managing validation processes.
    view_models : ViewModels
        View for managing machine learning models.
    view_metrics : ViewMetrics
        View for managing metrics display.
    view_results : ViewResults
        View for displaying results of models and metrics.
    vistas : list
        List containing all views in order corresponding to stacked widget indices.

    Methods
    -------
    connect_buttons()
        Connects toggle signals of navigation buttons to the view switching logic.
    _change_view(index_destino: int, checked: bool)
        Changes the displayed view to the given index if the button is checked and current view's data is valid.
    _sync_buttons(index: int)
        Updates the checked state of navigation buttons to reflect the current view.
    """
    def __init__(self):
        """
        Initialize the MainWindow, setup UI, create and add views to the stacked widget,
        initialize navigation buttons, and connect their signals.
        """
        super(MainWindow, self).__init__()

        # Load and apply UI to the main window
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Instantiate views and add them to the stacked widget for navigation
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

        # List of views to simplify access by index
        self.views = [
            self.view_datasets,      # index  0
            self.view_validation,    # index  1
            self.view_models,        # index  2
            self.view_metrics,       # index  3
            self.view_results        # index  4
        ]

        # Initial UI state setup
        self.ui.icon_only_widget.hide()
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.datasets_btn_1.setChecked(True)
        self.ui.datasets_btn_2.setChecked(True)

        # Connect navigation buttons to handlers
        self.connect_buttons()

    def connect_buttons(self):
        """
        Connect toggle signals of navigation buttons for all views to the handler
        that changes views when toggled and validates current view data.
        """
        buttons_by_index = {
            0: [self.ui.datasets_btn_1, self.ui.datasets_btn_2],
            1: [self.ui.val_btn_1, self.ui.val_btn_2],
            2: [self.ui.model_btn_1, self.ui.model_btn_2],
            3: [self.ui.metrics_btn_1, self.ui.metrics_btn_2],
            4: [self.ui.results_btn_1, self.ui.results_btn_2],
        }

        for index, buttons in buttons_by_index.items():
            for btn in buttons:
                btn.toggled.connect(lambda checked, i=index: self._change_view(i, checked))

    def _change_view(self, target_index, checked):
        """
        Change the current view in the stacked widget if the button is checked and
        the current view data is valid.

        Parameters
        ----------
        target_index : int
            The index of the destination view in the stacked widget.
        checked : bool
            Whether the toggle button was checked.
        """
        if not checked:
            return

        current_index = self.ui.stackedWidget.currentIndex()
        if current_index == target_index:
            return

        current_view = self.views[current_index]
        if hasattr(current_view, '_validate_data'):
            if not current_view._validate_data():
                self._sync_buttons(current_index)
                return

        self.ui.stackedWidget.setCurrentIndex(target_index)
        self._sync_buttons(target_index)


    def _sync_buttons(self, index):
        """
        Synchronize the checked state of navigation buttons to reflect the currently active view.

        Parameters
        ----------
        index : int
            The index of the currently active view.
        """
        buttons = {
            0: [self.ui.datasets_btn_1, self.ui.datasets_btn_2],
            1: [self.ui.val_btn_1, self.ui.val_btn_2],
            2: [self.ui.model_btn_1, self.ui.model_btn_2],
            3: [self.ui.metrics_btn_1, self.ui.metrics_btn_2],
            4: [self.ui.results_btn_1, self.ui.results_btn_2],
        }

        # Uncheck all buttons without emitting signals
        for grupo in buttons.values():
            for btn in grupo:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)

        # Check only the buttons corresponding to the active index without 
        # emitting signals
        for btn in buttons[index]:
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)

