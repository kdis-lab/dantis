from PyQt5.QtWidgets import QWidget, QStackedWidget, QScrollArea, QVBoxLayout
from PyQt5.QtWidgets import QHBoxLayout, QComboBox, QPushButton, QSizePolicy
from PyQt5.QtCore import Qt

from controller.model_controller import ModelController
from view.dialogs.dialogLoadModel import DialogLoadModel
from view.dialogs.dialogUpdateHyperparameters import DialogUpdateHyperparameters
from core.utils import show_error
from itertools import combinations
import logging


class ViewModels(QWidget):
    """
    View class for managing machine learning models within the application.

    This class allows users to add, modify, load, and delete models, 
    manage hyperparameters, and control navigation to the metrics view.

    Parameters
    ----------
    main_window : QMainWindow
        Reference to the main application window.
    ui : Ui_MainWindow
        The main user interface object containing widgets.
    DatasetController : DatasetController, optional
        Controller responsible for dataset management, by default None.
    ValidationController : ValidationController, optional
        Controller responsible for validation management, by default None.

    Attributes
    ----------
    ui : Ui_MainWindow
        Reference to the UI object.
    main_window : QMainWindow
        Reference to the main window.
    datasetController : DatasetController or None
        Dataset controller instance.
    validationController : ValidationController or None
        Validation controller instance.
    layout_counter : int
        Counter to uniquely identify dynamically added model layouts.
    modelController : ModelController
        Controller handling model-related business logic.
    nameAndType_models : dict
        Dictionary mapping model types to lists of model names.
    nameAndHyperparameters_models : dict
        Dictionary mapping model names to their hyperparameters.
    modelTypes : list
        List of available model types.
    selectedModels : dict
        Dictionary storing selected models and their hyperparameters keyed by layout identifier.

    Methods
    -------
    init_ui()
        Connects UI buttons to their respective event handlers.
    on_next_model_click()
        Handles the "Next" button click event and validates model data.
    check_info_model()
        Validates model information and enables navigation buttons.
    on_add_model_click()
        Adds a new model selection section with widgets and buttons.
    load_model()
        Opens a dialog to load a previously saved model and updates UI accordingly.
    update_model_combo_type(selected_type, identifier, combo_box_names)
        Updates model name combo box based on the selected model type.
    update_hiperparameters_type(identifier, combo_box_names)
        Updates hyperparameters for the selected model.
    update_model_combo_names(identifier, combo_box_names)
        Updates hyperparameters when the selected model changes.
    modify_model()
        Opens a dialog to modify hyperparameters of the selected model.
    delete_model_section()
        Deletes a model section from the UI and internal state.
    _validate_data()
        Validates that there are no duplicate models added.
    """
    def __init__(self, main_window, ui, DatasetController=None, ValidationController=None):
        """
        Initializes the ViewModels instance and loads available models.

        Parameters
        ----------
        main_window : QMainWindow
            Reference to the main window.
        ui : Ui_MainWindow
            Main UI object.
        DatasetController : DatasetController, optional
            Dataset controller.
        ValidationController : ValidationController, optional
            Validation controller.
        """
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
        """
        Connect UI buttons to their corresponding event handlers.
        """
        self.ui.add_model.clicked.connect(self.on_add_model_click)
        self.ui.next_model.clicked.connect(self.on_next_model_click)

    def on_next_model_click(self):
        """
        Handle click event of the 'Next' button by validating the current model info.
        """
        self._check_info_model()
    
    def _check_info_model(self):
        """
        Validate that the model data is correct and enable navigation buttons.
        """
        if self._validate_data():
            self.ui.metrics_btn_2.setDisabled(False)
            self.ui.metrics_btn_1.setDisabled(False)
            self.ui.metrics_btn_2.setChecked(True)
    
    def on_add_model_click(self):
        """
        Add a new model selection section with widgets and action buttons.
        """
        stacked_widget = self.main_window.findChild(QStackedWidget, "stackedWidget")
        identifier = self.layout_counter
        self.layout_counter += 1

        if stacked_widget is None or stacked_widget.count() <= 1:
            logging.error("Error: No se pudo encontrar stackedWidget o la página 1")
            return

        page_2 = stacked_widget.widget(2)
        scroll_area = page_2.findChild(QScrollArea, "scrollArea")
        
        if scroll_area is None:
            logging.error("Error: No se pudo encontrar el QScrollArea")
            return

        scroll_area_widget = scroll_area.widget()
        scroll_area_widget.setMinimumWidth(0)
        layout = scroll_area_widget.layout()

        if layout is None:
            layout = QVBoxLayout()
            scroll_area_widget.setLayout(layout) 
        layout.setAlignment(Qt.AlignTop)   
        
        horizontal_layout = QHBoxLayout()

        combo_box_type = QComboBox()
        combo_box_type.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo_box_type.addItems(self.modelTypes)

        combo_box_names = QComboBox()
        combo_box_names.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo_box_type.currentTextChanged.connect(lambda text: self.update_model_combo_type(text, identifier, combo_box_names))

        self.update_model_combo_type(combo_box_type.currentText(), identifier, combo_box_names)
        combo_box_names.currentTextChanged.connect(lambda: self.update_model_combo_names(identifier, combo_box_names))
        
        btn_modify = QPushButton("Modificar Hiperparámetros")
        btn_modify.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_load_model = QPushButton("Cargar Modelo")
        btn_load_model.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_delete = QPushButton("Eliminar")
        btn_delete.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        btn_modify.setProperty("identifier", identifier)
        btn_modify.setProperty("combo_box", combo_box_names)
        btn_delete.setProperty("identifier", identifier)
        btn_delete.setProperty("layout", horizontal_layout)

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
        layout.addSpacing(20)

        model_name = combo_box_names.currentText()
        hyperparameters = self.nameAndHyperparameters_models.get(model_name, {})

        self.selectedModels[identifier] = {
            "model": model_name,
            "hyperparameters": hyperparameters
        }

    def load_model(self):
        """
        Open a dialog to load a previously saved model, update UI and internal state accordingly.
        """
        sender = self.sender()
        identifier = sender.property("identifier") if sender else self.layout_counter - 1

        preloaded_paths = {"full": {}, "split": {}}
        if identifier in self.modelController.models and "paths" in self.modelController.models[identifier]:
            preloaded_paths = self.modelController.models[identifier]["paths"]

        dialog = DialogLoadModel(self, self.modelController, self.datasetController, identifier, preloaded_paths)

        if dialog.exec() == dialog.Accepted:
            if sender is not None:
                btn_modify = sender.property("btn_modify")
                combo_box_type = sender.property("combo_box_type")
                combo_box_names = sender.property("combo_box_names")

                if btn_modify:
                    btn_modify.setDisabled(True)
                if combo_box_type:
                    combo_box_type.setDisabled(True)
                if combo_box_names:
                    combo_box_names.setDisabled(True)

            self.selectedModels[identifier] = {
                "model": self.modelController.models[identifier]["model"],
                "hyperparameters": {},
                "paths": dialog.dataset_paths
            }
            self.modelController.add_model(identifier, self.selectedModels[identifier])

    def update_model_combo_type(self, selected_type, identifier, combo_box_names):
        """
        Update the model names combo box when the model type selection changes.

        Parameters
        ----------
        selected_type : str
            The selected model type.
        identifier : int
            Unique identifier for the model section.
        combo_box_names : QComboBox
            Combo box widget for model names.
        """
        combo_box_names.clear()
        models = self.nameAndType_models.get(selected_type, [])
        combo_box_names.addItems(models)
        self.update_hyperparameters_type(identifier, combo_box_names)

    def update_hyperparameters_type(self, identifier, combo_box_names): 
        """
        Update hyperparameters for the selected model.

        Parameters
        ----------
        identifier : int
            Unique identifier for the model section.
        combo_box_names : QComboBox
            Combo box widget with the selected model.
        """
        model_name = combo_box_names.currentText()
        hyperparameters = self.nameAndHyperparameters_models.get(model_name, {})
        self.selectedModels[identifier] = {
            "model": model_name,
            "hyperparameters": hyperparameters
        }
        self.modelController.add_model(identifier, self.selectedModels[identifier])

    def update_model_combo_names(self, identifier, combo_box_names):
        """
        Update hyperparameters when the selected model changes.

        Parameters
        ----------
        identifier : int
            Unique identifier for the model section.
        combo_box_names : QComboBox
            Combo box widget with the selected model.
        """
        self.update_hyperparameters_type(identifier, combo_box_names)

    def modify_model(self):
        """
        Open a dialog to modify hyperparameters of the selected model.
        """
        sender = self.sender()
        identifier = sender.property("identifier")
        combo_box_names = sender.property("combo_box")

        model = combo_box_names.currentText()
        if (identifier in self.modelController.models) and (self.modelController.models[identifier]['model'] == model):
            model = self.modelController.models[identifier]["model"]
            hyperparams = self.modelController.models[identifier]["hyperparameters"]
            dialog = DialogUpdateHyperparameters(self, self.modelController, identifier, model, hyperparams)
            dialog.exec_()

    def delete_model_section(self):
        """
        Delete a model section from the UI and internal state.
        """
        sender = self.sender()
        identifier = sender.property("identifier")
        horizontal_layout = sender.property("layout")

        scroll_area = self.ui.scrollArea
        scroll_area_widget = scroll_area.widget()
        main_layout = scroll_area_widget.layout()
        if main_layout is None:
            return

        index_to_remove = -1
        for i in range(main_layout.count()):
            item = main_layout.itemAt(i)
            if item.layout() == horizontal_layout:
                index_to_remove = i
                break

        if index_to_remove == -1:
            return

        for i in reversed(range(horizontal_layout.count())):
            widget = horizontal_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        item = main_layout.takeAt(index_to_remove)
        del item 

        if index_to_remove < main_layout.count():
            spacing_item = main_layout.itemAt(index_to_remove)
            if spacing_item.spacerItem():
                main_layout.takeAt(index_to_remove)
                del spacing_item 

        self.modelController.delete_model(identifier)

        self.layout_counter = max(self.modelController.models.keys()) + 1 if self.modelController.models else 0

        scroll_area_widget.updateGeometry()
        scroll_area.update()
    
    def _validate_data(self): 
        """
        Validate that no duplicate models have been added.

        Returns
        -------
        bool
            True if all models are unique; False otherwise.
        """
        content = self.ui.scrollArea.widget()

        if content is not None and content.findChildren(QWidget):
            models = self.modelController.models
            
            def equal_models(m1, m2):
                if m1['model'] != m2['model']:
                    return False
                return m1['hyperparameters'] == m2['hyperparameters']

            ids = list(models.keys())
            for id1, id2 in combinations(ids, 2):
                if equal_models(models[id1], models[id2]):
                    name_model = models[id1]['model']
                    show_error(f"¡Atención! Los models '{name_model}' son idénticos (nombre y parámetros). Por favor elimina uno.")
                    return False

            return True
        else:
            show_error("Para poder continuar, debe agregar algún modelo para entrenar los datos previamente introducidos.")
            return False

    def resizeEvent(self, event):
        """
        Override of the QWidget resize event to adjust the scroll area's content width.

        Ensures that the inner widget of the scroll area dynamically resizes 
        to match the visible viewport width whenever the main window is resized.

        Parameters
        ----------
        event : QResizeEvent
            The event object containing the new size and old size.
        """
        super().resizeEvent(event)

        stacked_widget = self.main_window.findChild(QStackedWidget, "stackedWidget")
        if stacked_widget and stacked_widget.count() > 1:
            page_2 = stacked_widget.widget(2)
            scroll_area = page_2.findChild(QScrollArea, "scrollArea")
            if scroll_area:
                scroll_widget = scroll_area.widget()
                if scroll_widget:
                    scroll_widget.setMinimumWidth(scroll_area.viewport().width())
                    scroll_widget.updateGeometry()