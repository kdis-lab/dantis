
from PyQt5.QtWidgets import QVBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox
from PyQt5.QtWidgets import QComboBox, QCheckBox, QLineEdit, QPushButton, QDialog
import ast

class DialogUpdateHiperparameters(QDialog):
    def __init__(self, parent=None, modelController=None, id=None, model=None, hiperparameters=None):
        super().__init__(parent)
        self.setWindowTitle(f"Editar hiperparámetros - {model}")
        self.parent_window = parent  # Referencia a la ventana principal
        self.modelController = modelController
        self.id = id
        self.model = model
        self.fields = {}
        self.hiperparameters = hiperparameters
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Crear dinámicamente los campos de los hiperparámetros
        for param, value in self.hiperparameters.items():
            widget = self._create_widget_for_value(value)
            self.fields[param] = widget
            form_layout.addRow(param, widget)

        layout.addLayout(form_layout)

        # Botón de guardar
        btn_guardar = QPushButton("Guardar Hiperparámetros")
        btn_guardar.clicked.connect(self.save_hiperparameters)
        layout.addWidget(btn_guardar)

        self.setLayout(layout)

    def _create_widget_for_value(self, value):
        """Crea el widget adecuado según el tipo del valor."""
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
            widget.setText(str(value))  # Fallback
        return widget
    
    def save_hiperparameters(self):
        """Obtiene los valores actuales de los widgets y los guarda."""
        updated_params = {}
        for param, widget in self.fields.items():
            if isinstance(widget, QCheckBox):
                updated_params[param] = widget.isChecked()
            elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                updated_params[param] = widget.value()
            elif isinstance(widget, QLineEdit):
                text = widget.text()
                try:
                    # Intenta convertir el texto a su tipo original (list, None, float, etc.)
                    parsed_value = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    # Si no se puede evaluar, lo deja como texto plano
                    parsed_value = text
                updated_params[param] = parsed_value
            elif isinstance(widget, QComboBox):
                updated_params[param] = widget.currentText()
        
        hiperparametersModel = {}
        hiperparametersModel[self.id] = {
            "model": self.model,
            "hyperparameters": updated_params
        }

        # Opcional: Actualizar en el controlador directamente
        if self.modelController:
            self.modelController.update_model(self.id, hiperparametersModel[self.id])

        self.accept()  # Cierra el diálogo


    """
    INFORMACIÓN IMPORTANTE PARA CUANDO SE VAYAN A MODIFICAR LOS HIPERPARAMETROS
    TOMAR ESTA INFORMACIÓN COMO REFERENCIA

        # Crear los controles para cada parámetro en el diccionario
        for key, value in self.updateHyperparameters.items():
            if key in config_data:
                param_config = config_data[key]  # Obtener las opciones a modificar del JSON
                
                if param_config["type"] == "int" and "options" in param_config:
                    combo_box = QComboBox()
                    combo_box.addItems(param_config["options"])

                    if value in param_config["options"]:
                        combo_box.setCurrentText(value)

                    form_layout.addRow(str(key), combo_box)
                    self.inputs[key] = (combo_box, "combo")
                    continue

                elif param_config["type"] == "str" and "options" in param_config:
                    combo_box = QComboBox()

                    # Agregar las opciones al combo box
                    combo_box.addItems(param_config["options"])

                    if value is None:
                        combo_box.setCurrentText("None")
                    else:
                        combo_box.setCurrentText(value)

                    # Añadir el combo box al formulario
                    form_layout.addRow(str(key), combo_box)
                    self.inputs[key] = (combo_box, "combo")
                    continue

            else:
                if isinstance(value, bool):
                    checkbox = QCheckBox(str(key))
                    checkbox.setChecked(value)
                    form_layout.addRow(str(key), checkbox)
                    self.inputs[key] = (checkbox, "check")
                elif isinstance(value, int):
                    line_edit_int = QLineEdit(str(value))
                    form_layout.addRow(str(key), line_edit_int)
                    self.inputs[key] = (line_edit_int, "text")
                elif isinstance(value, float):
                    line_edit_float = QLineEdit(str(value))
                    form_layout.addRow(str(key), line_edit_float)
                    self.inputs[key] = (line_edit_float, "text")
                elif isinstance(value, list):
                    line_edit_list = QLineEdit()
                    # Convertir la lista en una cadena separada por comas
                    line_edit_list.setText(", ".join(map(str, value)))
                    form_layout.addRow(str(key), line_edit_list)
                    self.inputs[key] = (line_edit_list, "list")
                elif value is None:
                    line_edit = QLineEdit(str(value))
                    form_layout.addRow(str(key), line_edit)
                    self.inputs[key] = (line_edit, "text")
                else: 
                    line_edit = QLineEdit(str(value))
                    form_layout.addRow(str(key), line_edit)
                    self.inputs[key] = (line_edit, "text")

    """
        
    """
    def save_changes(self):
        # Guardar los cambios en el diccionario
        for key, (widget, input_type) in self.inputs.items():
            if isinstance(widget, QCheckBox):
                self.updateHyperparameters[key] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                self.updateHyperparameters[key] = widget.value()  ## NO SE USA
            elif isinstance(widget, QDoubleSpinBox):
                self.updateHyperparameters[key] = widget.value()  ## NO SE USA
            elif isinstance(widget, QComboBox):
                self.updateHyperparameters[key] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                input_text = widget.text()
                if input_type == "list":
                    try:
                        if "," in input_text: 
                            self.updateHyperparameters[key] = list(map(int, input_text.split(", ")))
                        else:  # Lista con un solo elemento
                            self.updateHyperparameters[key] = [int(input_text)]
                    except ValueError:
                        # Si no puede convertirse a enteros, almacenar como lista de texto
                        if "," in input_text:
                            self.updateHyperparameters[key] = input_text.split(", ")
                        else:
                            self.updateHyperparameters[key] = [input_text]
                elif input_type == "text":
                    # Procesar como número o texto
                    try:
                        # Intentar convertir primero a float
                        self.updateHyperparameters[key] = float(input_text)
                    except ValueError:
                        # Si falla, almacenarlo como texto
                        self.updateHyperparameters[key] = input_text
        self.accept()
    """