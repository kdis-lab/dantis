from PyQt5.QtWidgets import QWidget, QCheckBox

from controller.metrics_controller import MetricsController
from core.utils import show_error

class ViewMetrics(QWidget):
    """
    ViewMetrics class handles the metrics selection interface and logic.

    This view manages the metric checkboxes, validates that at least one metric is selected,
    and controls the navigation to the results view when the user proceeds.

    Attributes
    ----------
    ui : Ui_MainWindow
        The main UI object containing widgets.
    main_window : QMainWindow
        Reference to the main window.
    metricsController : MetricsController
        Controller responsible for managing metrics data and state.
    checkboxes : list of QCheckBox
        List of checkbox widgets found in the metrics page.

    Methods
    -------
    init_ui()
        Connects the "next" button click to the metrics check handler.
    on_next_metrics_click()
        Triggered when the user clicks the next button to validate and proceed.
    checkBoxes_selected() -> bool
        Returns True if any of the metric checkboxes are checked.
    get_checked_checkboxes() -> list
        Returns a list of texts for all checked metric checkboxes.
    _check_info_metrics()
        Validates the metrics data and enables navigation buttons if valid.
    _validate_data() -> bool
        Performs validation to ensure at least one metric is selected,
        shows an error otherwise.
    """
    def __init__(self, main_window, ui):
        """
        Initialize the ViewMetrics instance, setting up UI references and controller.
        
        Parameters
        ----------
        main_window : QMainWindow
            Reference to the main application window.
        ui : Ui_MainWindow
            The UI object containing widgets.
        """
        super().__init__()
        self.ui = ui
        self.main_window = main_window
        self.metricsController = MetricsController()
        self.init_ui()

    def init_ui(self):
        """
        Connect the 'next_metrics' button's clicked signal to the handler method.
        """
        self.ui.next_metrics.clicked.connect(self.on_next_metrics_click)

    def on_next_metrics_click(self):
        """
        Handler called when the 'next_metrics' button is clicked.
        Validates metrics info before proceeding.
        """
        self._check_info_metrics()

    def _checkBoxes_selected(self):
        """
        Check if any metric checkbox is selected.

        Returns
        -------
        bool
            True if at least one checkbox is checked, False otherwise.
        """
        return any(cb.isChecked() for cb in self.checkboxes)
    
    def _get_checked_checkboxes(self):
        """
        Get the list of labels/texts of all checked metric checkboxes.

        Returns
        -------
        list of str
            Text labels of the checked checkboxes.
        """
        return [cb.text() for cb in self.checkboxes if cb.isChecked()]

    def _check_info_metrics(self):
        """
        Validate metric selections and if valid, enable navigation buttons and
        switch to the results view.
        """
        if self._validate_data(): 
            self.ui.results_btn_2.setDisabled(False)
            self.ui.results_btn_1.setDisabled(False)
            self.ui.results_btn_2.setChecked(True)

    def _validate_data(self): 
        """
        Validate that at least one metric checkbox is selected.

        Returns
        -------
        bool
            True if valid (at least one selected), False otherwise.
        """
        page3_widget = self.ui.stackedWidget.widget(3)
        self.checkboxes = page3_widget.findChildren(QCheckBox)

        self.metricsController.setChecBoxesSelected(self._get_checked_checkboxes())

        if self._checkBoxes_selected():
            return True
        else:
            show_error("Debe seleccionar al menos una métrica.")
            return False