from PyQt5.QtWidgets import QVBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox
from PyQt5.QtWidgets import QComboBox, QCheckBox, QLineEdit, QPushButton, QDialog
import ast

class DialogUpdateHyperparameters(QDialog):
    """
    Dialog window to dynamically edit hyperparameters of a given model.

    This dialog creates input widgets based on the type of each hyperparameter
    and allows users to modify and save them. It supports common types like bool,
    int, float, str, and lists of strings, using appropriate Qt widgets.

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget of the dialog. Default is None.
    modelController : object, optional
        Controller to manage model updates. Default is None.
    id : int or str, optional
        Identifier for the model or context. Default is None.
    model : str, optional
        Name of the model being edited. Used in the window title. Default is None.
    hyperparameters : dict, optional
        Dictionary of hyperparameters and their current values.

    Attributes
    ----------
    parent_window : QWidget
        Reference to the parent widget.
    modelController : object
        Controller handling model updates.
    id : int or str
        Model or context identifier.
    model : str
        Model name.
    fields : dict
        Dictionary mapping hyperparameter names to their corresponding input widgets.
    hyperparameters : dict
        Original hyperparameters to be edited.
    """
    def __init__(self, parent=None, modelController=None, id=None, model=None, hyperparameters=None):
        """
        Initialize the dialog window and its UI components.

        Sets the dialog title, stores references to controller and parameters,
        and calls the UI initialization method.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        modelController : object, optional
            Controller for model updates.
        id : int or str, optional
            Model or context identifier.
        model : str, optional
            Model name.
        hyperparameters : dict, optional
            Dictionary of hyperparameters and their values.
        """
        super().__init__(parent)
        self.setWindowTitle(f"Editar hiperparámetros - {model}")
        self.parent_window = parent  # Referencia a la ventana principal
        self.modelController = modelController
        self.id = id
        self.model = model
        self.fields = {}
        self.hyperparameters = hyperparameters
        self.init_ui()
        
    def init_ui(self):
        """
        Set up the UI layout with form inputs corresponding to each hyperparameter.

        Dynamically creates widgets for each parameter using `_create_widget_for_value`.
        Adds a "Guardar Hiperparámetros" (Save Hyperparameters) button that triggers saving.
        """
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        for param, value in self.hyperparameters.items():
            widget = self._create_widget_for_value(value)
            self.fields[param] = widget
            form_layout.addRow(param, widget)

        layout.addLayout(form_layout)

        btn_save = QPushButton("Guardar Hiperparámetros")
        btn_save.clicked.connect(self.save_hyperparameters)
        layout.addWidget(btn_save)

        self.setLayout(layout)

    def _create_widget_for_value(self, value):
        """
        Create an appropriate input widget for a given hyperparameter value type.

        Parameters
        ----------
        value : bool, int, float, str, list or other
            The current value of the hyperparameter.

        Returns
        -------
        QWidget
            A Qt widget suitable for editing the type of `value`.
        """
        ...
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
        elif isinstance(value, int):
            widget = QSpinBox()
            widget.setMaximum(999999)
            widget.setValue(value)
        elif isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setDecimals(4)
            widget.setMaximum(1e6)
            widget.setValue(value)
        elif isinstance(value, str):
            widget = QLineEdit()
            widget.setText(value)
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            widget = QComboBox()
            widget.addItems(value)
        else:
            widget = QLineEdit()
            widget.setText(str(value))
        return widget

    def save_hyperparameters(self):
        """
        Collect the updated hyperparameter values from input widgets and save them.

        Parses inputs back to their original types where possible. Calls
        `update_model` on the modelController with the updated parameters if available,
        then closes the dialog with acceptance.
        """
        updated_params = {}
        for param, widget in self.fields.items():
            if isinstance(widget, QCheckBox):
                updated_params[param] = widget.isChecked()
            elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                updated_params[param] = widget.value()
            elif isinstance(widget, QLineEdit):
                text = widget.text()
                try:
                    parsed_value = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed_value = text
                updated_params[param] = parsed_value
            elif isinstance(widget, QComboBox):
                updated_params[param] = widget.currentText()

        hyperparametersModel = {}
        hyperparametersModel[self.id] = {
            "model": self.model,
            "hyperparameters": updated_params
        }

        if self.modelController:
            self.modelController.update_model(self.id, hyperparametersModel[self.id])

        self.accept() 