from PyQt5.QtWidgets import QWidget, QStackedWidget, QScrollArea, QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout, QComboBox, QPushButton
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import Qt
from view.dialogs.dialogUpdateHiperparameter import DialogUpdateHiperparameters

from controller.model_controller import ModelController
from view.dialogs.dialogLoadModel import DialogLoadModel

from core.utils import show_error

class ViewModels(QWidget):
    def __init__(self, main_window, ui, DatasetController=None, ValidationController=None):
        super().__init__()
        self.ui = ui
        self.main_window = main_window
        self.datasetController = DatasetController
        self.validationController = ValidationController
        self.layout_counter = 0
        self.modelController = ModelController()
        self.nameAndType_models, self.nameAndHyperparameters_models = self.modelController.get_available_models()
        self.modelTypes = list(self.nameAndType_models.keys())
        self.selectedModels = {}
        self.init_ui()

    def init_ui(self):
        self.ui.add_model.clicked.connect(self.on_add_model_click)
        self.ui.next_model.clicked.connect(self.on_next_model_click)

    def on_next_model_click(self):
        self.check_info_model()
    
    def check_info_model(self):

        if self.validar_datos(): 
            self.ui.metrics_btn_2.setDisabled(False)
            self.ui.metrics_btn_1.setDisabled(False)

            # Para cambiar a la página de los modelos
            self.ui.metrics_btn_2.setChecked(True)
    
    def on_add_model_click(self):
        # Accede directamente al stackedWidget si está dentro de la clase MainWindow
        stacked_widget = self.main_window.findChild(QStackedWidget, "stackedWidget")

        # Genera un identificador único basado en el contador
        identifier = self.layout_counter
        self.layout_counter += 1

        # Verifica que stacked_widget y la página existen antes de proceder
        if stacked_widget is None or stacked_widget.count() <= 1:
            print("Error: No se pudo encontrar stackedWidget o la página 1")
            return

        # Accede a la página con índice 2 en el stackedWidget
        page_2 = stacked_widget.widget(2)

        # Encuentra el QScrollArea dentro de la página 2
        scroll_area = page_2.findChild(QScrollArea, "scrollArea")
        
        if scroll_area is None:
            print("Error: No se pudo encontrar el QScrollArea")
            return

        # Accede al widget del scroll area, que es donde se colocarán los widgets
        scroll_area_widget = scroll_area.widget()
        scroll_area_widget.setMinimumWidth(0)

        # Verifica si el layout del widget en el scroll area ya está configurado
        layout = scroll_area_widget.layout()

        if layout is None:
            layout = QVBoxLayout()
            layout.setAlignment(Qt.AlignTop)
            scroll_area_widget.setLayout(layout) 
            
        
        # Crea un QHBoxLayout para contener los ComboBox y el botón
        horizontal_layout = QHBoxLayout()

        # Crea el primer combo box. Tipos de modelos [deep Learning, Machine Learning, Statistical]
        combo_box_type = QComboBox()
        combo_box_type.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        #combo_box_type.setObjectName("combo_box_type")
        combo_box_type.addItems(self.modelTypes)

        # Crea el segundo combo box. Los modelos correspondientes al tipo de modelo seleccionado previamente.
        combo_box_names = QComboBox()
        combo_box_names.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        #combo_box_names.setObjectName("combo_box_names")
        combo_box_type.currentTextChanged.connect(lambda texto: self.update_model_combo_type(texto, identifier, combo_box_names))
        self.update_model_combo_type(combo_box_type.currentText(), identifier, combo_box_names)
        combo_box_names.currentTextChanged.connect(lambda: self.update_model_combo_names(identifier, combo_box_names))
        
        btn_modify = QPushButton("Modificar Hiperparámetros")
        btn_modify.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_load_model = QPushButton("Cargar Modelo")
        btn_load_model.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_delete = QPushButton("Eliminar")
        btn_delete.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        ## ESTAS 4 LINEAS SE PODRIAN BORRAR? 
        #############################
        btn_modify.setProperty("identifier", identifier)
        btn_modify.setProperty("combo_box", combo_box_names)
        btn_delete.setProperty("identifier", identifier)
        btn_delete.setProperty("layout", horizontal_layout)
        #############################

        btn_modify.clicked.connect(self.modify_model)
        btn_load_model.setProperty("identifier", identifier)
        btn_load_model.setProperty("combo_box_names", combo_box_names)
        btn_load_model.setProperty("btn_modify", btn_modify)
        btn_load_model.setProperty("combo_box_type", combo_box_type)
        btn_load_model.clicked.connect(self.load_model)
        btn_delete.clicked.connect(self.delete_model_section)

        horizontal_layout.addWidget(combo_box_type, 3)
        horizontal_layout.addWidget(combo_box_names, 3)
        horizontal_layout.addWidget(btn_modify, 2)
        horizontal_layout.addWidget(btn_load_model, 2)
        horizontal_layout.addWidget(btn_delete, 2)
        layout.addLayout(horizontal_layout)

        #Guardamos el identificador, el modelo y los hiperparametros del mismo en el diccionario
        nombre_modelo = combo_box_names.currentText()
        hiperparametros = self.nameAndHyperparameters_models.get(nombre_modelo, {})

        self.selectedModels[identifier] = {
            "model": nombre_modelo,
            "hyperparameters": hiperparametros
        }
        #self.modelController.add_model(identifier, self.selectedModels[identifier])

    def load_model(self):
        sender = self.sender()
        identifier = sender.property("identifier") if sender else self.layout_counter - 1

        # Obtener paths previamente guardados desde selectedModels
        preloaded_paths = {"full": {}, "split": {}}
        if identifier in self.modelController.models and "paths" in self.modelController.models[identifier]:
            preloaded_paths = self.modelController.models[identifier]["paths"]

        dialog = DialogLoadModel(self, self.modelController, self.datasetController, identifier, preloaded_paths)

        if dialog.exec_() == dialog.Accepted:
            # Bloquear el botón de modificar hiperparámetros
            if sender is not None:
            # Desactivar todos los elementos de selección
                btn_modify = sender.property("btn_modify")
                combo_box_type = sender.property("combo_box_type")
                combo_box_names = sender.property("combo_box_names")

                if btn_modify:
                    btn_modify.setDisabled(True)
                if combo_box_type:
                    combo_box_type.setDisabled(True)
                if combo_box_names:
                    combo_box_names.setDisabled(True)
            # Actualizar el modelo seleccionado para que no tenga hiperparámetros
            self.selectedModels[identifier] = {
                "model": self.modelController.models[identifier]["model"],
                "hyperparameters": {},
                "paths": dialog.dataset_paths
            }
            self.modelController.add_model(identifier, self.selectedModels[identifier])

    def update_model_combo_type(self, tipo_seleccionado, identifier, combo_box_names):
        combo_box_names.clear()
        modelos = self.nameAndType_models.get(tipo_seleccionado, [])
        combo_box_names.addItems(modelos)
        self.update_hiperparameters_type(identifier, combo_box_names)

    def update_hiperparameters_type(self, identifier, combo_box_names): 
        nombre_modelo = combo_box_names.currentText()
        hiperparametros = self.nameAndHyperparameters_models.get(nombre_modelo, {})
        self.selectedModels[identifier] = {
            "model": nombre_modelo,
            "hyperparameters": hiperparametros
        }
        self.modelController.add_model(identifier, self.selectedModels[identifier])

    def update_model_combo_names(self, identifier, combo_box_names):
        self.update_hiperparameters_type(identifier, combo_box_names)

    def modify_model(self):
        sender = self.sender()
        identifier = sender.property("identifier")
        combo_box_names = sender.property("combo_box")

        modelo = combo_box_names.currentText()
        if (identifier in self.modelController.models) and (self.modelController.models[identifier]['model'] == modelo):
            modelo = self.modelController.models[identifier]["model"]
            hyperparams = self.modelController.models[identifier]["hyperparameters"]
            dialog = DialogUpdateHiperparameters(self, self.modelController, identifier, modelo, hyperparams)
            dialog.exec_()

    def delete_model_section(self):
        sender = self.sender()
        identifier = sender.property("identifier")
        horizontal_layout = sender.property("layout")
        # Elimina todos los widgets del layout horizontal
        
        for i in reversed(range(horizontal_layout.count())):
            widget = horizontal_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        self.modelController.delete_model(identifier)

        self.layout_counter = max(self.modelController.models.keys()) + 1 if self.modelController.models else 0
    
    def validar_datos(self): 
        contenido = self.ui.scrollArea.widget()

        if contenido is not None and contenido.findChildren(QWidget):
            modelos = self.modelController.models
            from itertools import combinations

            def modelos_iguales(m1, m2):
                if m1['model'] != m2['model']:
                    return False
                return m1['hyperparameters'] == m2['hyperparameters']

            ids = list(modelos.keys())
            for id1, id2 in combinations(ids, 2):
                if modelos_iguales(modelos[id1], modelos[id2]):
                    nombre_modelo = modelos[id1]['model']
                    show_error(f"¡Atención! Los modelos '{nombre_modelo}' son idénticos (nombre y parámetros). Por favor elimina uno.")
                    return False

            return True
        else:
            show_error("Para poder continuar, debe agregar algún modelo para entrenar los datos previamente introducidos.")
            return False

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Ajusta el ancho mínimo del contenido del scrollArea
        stacked_widget = self.main_window.findChild(QStackedWidget, "stackedWidget")
        if stacked_widget and stacked_widget.count() > 1:
            page_2 = stacked_widget.widget(2)
            scroll_area = page_2.findChild(QScrollArea, "scrollArea")
            if scroll_area:
                scroll_widget = scroll_area.widget()
                if scroll_widget:
                    scroll_widget.setMinimumWidth(scroll_area.viewport().width())
                    scroll_widget.updateGeometry()