from PyQt5.QtWidgets import QWidget, QCheckBox

from core.utils import show_error

from controller.metrics_controller import MetricsController

class ViewMetrics(QWidget):
    def __init__(self, main_window, ui):
        super().__init__()
        self.ui = ui
        self.main_window = main_window
        self.metricsController = MetricsController()
        self.init_ui()

    def init_ui(self):
        self.ui.next_metrics.clicked.connect(self.on_next_metrics_click)

    def on_next_metrics_click(self):
        self.check_info_metrics()

    ## Funcion para comprobar si hay checkboxes marcadas
    def checkBoxes_selected(self):
        return any(cb.isChecked() for cb in self.checkboxes)
    
    def get_checked_checkboxes(self):
        return [cb.text() for cb in self.checkboxes if cb.isChecked()]

    def check_info_metrics(self):
        if self.validar_datos(): 
            self.ui.results_btn_2.setDisabled(False)
            self.ui.results_btn_1.setDisabled(False)

            # Para cambiar a la página de los modelos
            self.ui.results_btn_2.setChecked(True)

    def validar_datos(self): 
        page3_widget = self.ui.stackedWidget.widget(3)
        self.checkboxes = page3_widget.findChildren(QCheckBox)

        self.metricsController.setChecBoxesSelected(self.get_checked_checkboxes())
        # Puedes usar esto para comprobar si al menos uno está marcado
        if self.checkBoxes_selected():
            return True
        else:
            show_error("Debe seleccionar al menos una métrica.")
            return False