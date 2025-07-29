from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QHeaderView, QButtonGroup
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QPushButton

from core.utils import show_error
from controller.validation_controller import ValidationController
from view.widgets import DropListWidget, DropTableWidget

class ViewValidation(QWidget):
    """
    GUI component responsible for configuring dataset validation strategies.

    This view allows the user to choose between different validation types 
    (Train/Test, Cross-Validation, Sliding Split) and configure parameters 
    accordingly. It also enables dataset assignment to different roles 
    (train, validation, test) via drag-and-drop widgets.

    Parameters
    ----------
    main_window : QMainWindow
        Reference to the main application window.
    ui : object
        The UI object containing widgets defined in the Qt Designer form.
    DatasetController : DatasetController, optional
        Controller managing dataset operations and metadata, by default None.

    Attributes
    ----------
    validationController : ValidationController
        Handles validation configuration logic and persistence.
    datasetController : DatasetController
        Reference to the dataset manager.
    validation_config : dict
        Stores the current validation configuration to be passed to the controller.

    Methods
    -------
    init_ui()
        Sets up UI components, button bindings, and custom drag-and-drop widgets.
    on_delete_row_click()
        Deletes a dataset row and returns its contents to the available dataset list.
    initial_parameter_configuration()
        Initializes the list of datasets based on the controller's stored data.
    on_insert_row_click()
        Inserts a new empty row into the validation table with a delete button.
    on_radioButton_train_test_click()
        Enables UI for train/test validation configuration.
    on_radioButton_crossVal_click()
        Enables UI for cross-validation configuration.
    on_radioButton_sliding_click()
        Enables UI for sliding split configuration.
    on_next_val_click()
        Triggers validation check and advances if data is valid.
    _check_info_val()
        Main method to verify the correctness of the selected validation strategy.
    _validate_data()
        Performs thorough validation of selected datasets and parameter settings.
    on_radioButton_no_click()
        Disables manual dataset assignment when "No" is selected.
    on_radioButton_yes_click()
        Enables manual dataset assignment when "Yes" is selected.
    """
    def __init__(self, main_window, ui, DatasetController=None):
        """
        Initialize the ViewValidation component.

        Parameters
        ----------
        main_window : QMainWindow
            Main application window.
        ui : object
            User interface instance containing visual components.
        DatasetController : DatasetController, optional
            Dataset controller to retrieve and validate datasets.
        """
        super().__init__()
        self.main_window = main_window
        self.ui = ui
        self.validationController = ValidationController()
        self.datasetController = DatasetController
        self.validation_config = {}
        self.init_ui()

    def init_ui(self):
        """
        Sets up the user interface elements and binds UI events to handlers.

        This method also replaces the original list and table widgets with
        custom drag-and-drop versions (`DropListWidget`, `DropTableWidget`)
        and synchronizes UI layouts and policies accordingly.
        """
        self.group_validacion = QButtonGroup(self)
        self.group_validacion.addButton(self.ui.radioButton_train_test)
        self.group_validacion.addButton(self.ui.radioButton_crossVal)
        self.group_validacion.addButton(self.ui.radioButton_sliding)

        self.group_si_no = QButtonGroup(self)
        self.group_si_no.addButton(self.ui.radioButton_no)
        self.group_si_no.addButton(self.ui.radioButton_yes)

        self.ui.radioButton_train_test.setChecked(True)
        self.ui.radioButton_no.setChecked(True)
        self.ui.next_val.clicked.connect(self.on_next_val_click)

        self.ui.radioButton_train_test.clicked.connect(self.on_radioButton_train_test_click)
        self.ui.radioButton_crossVal.clicked.connect(self.on_radioButton_crossVal_click)
        self.ui.radioButton_sliding.clicked.connect(self.on_radioButton_sliding_click)
        self.ui.radioButton_no.clicked.connect(self.on_radioButton_no_click)
        self.ui.radioButton_yes.clicked.connect(self.on_radioButton_yes_click)
        self.ui.pushButton_insertRow.clicked.connect(self.on_insert_row_click)

        drop_list_widget = DropListWidget()
        drop_table_widget = DropTableWidget(drop_list_widget)

        layout = self.ui.horizontalLayout_listTabla
        layout.replaceWidget(self.ui.listWidget_datasets, drop_list_widget)
        layout.replaceWidget(self.ui.tableWidget_TVT, drop_table_widget)

        self.old_table = self.ui.tableWidget_TVT
        drop_table_widget.setColumnCount(self.old_table.columnCount())
        drop_table_widget.setHorizontalHeaderLabels(
            [self.old_table.horizontalHeaderItem(i).text() for i in range(self.old_table.columnCount())]
        )

        header = drop_table_widget.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        size_policy_list = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        drop_list_widget.setSizePolicy(size_policy_list)
        size_policy_table = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        drop_table_widget.setSizePolicy(size_policy_table)
        self.ui.listWidget_datasets.hide()
        self.ui.tableWidget_TVT.hide()

        self.ui.listWidget_datasets = drop_list_widget
        self.ui.tableWidget_TVT = drop_table_widget

        self.ui.listWidget_datasets.setDragEnabled(True)
        self.ui.tableWidget_TVT.setDragEnabled(True)
        self.ui.listWidget_datasets.setDisabled(True)
        self.ui.tableWidget_TVT.setDisabled(True)

        for i in range(self.ui.gridLayout_crossVal.count()):
            widget_crossval = self.ui.gridLayout_crossVal.itemAt(i).widget()
            widget_sliding = self.ui.gridLayout_sliding.itemAt(i).widget()
            if widget_crossval:
                widget_crossval.setDisabled(True)
            if widget_sliding:
                widget_sliding.setDisabled(True)

    def on_delete_row_click(self):
        """
        Removes the row associated with the clicked delete button and
        restores any dataset names to the list widget.
        """
        button = self.sender()
        if button:
            for row in range(self.ui.tableWidget_TVT.rowCount()):
                widget = self.ui.tableWidget_TVT.cellWidget(row, self.ui.tableWidget_TVT.columnCount() - 1)
                if widget == button:
                    for col in range(self.ui.tableWidget_TVT.columnCount() - 1):
                        item = self.ui.tableWidget_TVT.item(row, col)
                        if item:
                            self.ui.listWidget_datasets.addItem(item.text())
                    self.ui.tableWidget_TVT.removeRow(row)
                    break

    def initial_parameter_configuration(self):
        """
        Loads dataset names from the controller and populates the list widget.
        """
        datasets = self.datasetController.get_data()
        names = [dataset.name for dataset in datasets.values()]
        self.ui.listWidget_datasets.clear()
        self.ui.listWidget_datasets.addItems(names)

    def on_insert_row_click(self):
        """
        Inserts an empty row into the table and adds a delete button to the last column.
        """
        rowPosition = self.ui.tableWidget_TVT.rowCount()
        self.ui.tableWidget_TVT.insertRow(rowPosition)

        btn_borrar = QPushButton("🗑️")
        btn_borrar.clicked.connect(self.on_delete_row_click)

        self.ui.tableWidget_TVT.setCellWidget(rowPosition, self.ui.tableWidget_TVT.columnCount() - 1, btn_borrar)


    def on_radioButton_train_test_click(self):
        """
        Enables spin boxes for Train/Test configuration and disables others.
        """
        # ACTIVATE ELEMENTS FOR TRAIN/TEST
        self.ui.spinBox_train.setDisabled(False)
        self.ui.spinBox_val.setDisabled(False)
        self.ui.spinBox_test.setDisabled(False)

        # DISABLE ELEMENTS FOR CROSS VALIDATION
        self.ui.spinBox_crossVal.setValue(self.ui.spinBox_crossVal.minimum())
        self.ui.spinBox_crossVal.setDisabled(True)
        self.ui.spinBox_crossval_percent.setValue(self.ui.spinBox_crossval_percent.minimum())
        self.ui.spinBox_crossval_percent.setDisabled(True)

        # DISABLE ELEMENTS FOR SLIDING SPLIT
        self.ui.spinBox_sliding.setValue(self.ui.spinBox_sliding.minimum())
        self.ui.spinBox_sliding.setDisabled(True)
        self.ui.spinBox_sliding_percent.setValue(self.ui.spinBox_sliding_percent.minimum())
        self.ui.spinBox_sliding_percent.setDisabled(True)

    def on_radioButton_crossVal_click(self):
        """
        Enables Cross-Validation spin boxes and disables others.
        """
        # DISABLE ELEMENTS FOR TRAIN/TEST
        self.ui.spinBox_train.setValue(self.ui.spinBox_train.minimum())
        self.ui.spinBox_train.setDisabled(True)

        self.ui.spinBox_val.setValue(self.ui.spinBox_val.minimum())
        self.ui.spinBox_val.setDisabled(True)

        self.ui.spinBox_test.setValue(self.ui.spinBox_test.minimum())
        self.ui.spinBox_test.setDisabled(True)

        # ACTIVATE ELEMENTS FOR CROSS VALIDATION
        self.ui.spinBox_crossVal.setDisabled(False)
        self.ui.spinBox_crossval_percent.setDisabled(False)

        # DISABLE ELEMENTS FOR SLIDING SPLIT
        self.ui.spinBox_sliding.setValue(self.ui.spinBox_sliding.minimum())
        self.ui.spinBox_sliding.setDisabled(True)

        self.ui.spinBox_sliding_percent.setValue(self.ui.spinBox_sliding_percent.minimum())
        self.ui.spinBox_sliding_percent.setDisabled(True)

    def on_radioButton_sliding_click(self):
        """
        Enables Sliding Split spin boxes and disables others.
        """
        # DISABLE ELEMENTS FOR TRAIN/TEST
        self.ui.spinBox_train.setValue(self.ui.spinBox_train.minimum())
        self.ui.spinBox_train.setDisabled(True)

        self.ui.spinBox_val.setValue(self.ui.spinBox_val.minimum())
        self.ui.spinBox_val.setDisabled(True)

        self.ui.spinBox_test.setValue(self.ui.spinBox_test.minimum())
        self.ui.spinBox_test.setDisabled(True)

        # DISABLE ELEMENTS FOR CROSS VALIDATION
        self.ui.spinBox_crossVal.setValue(self.ui.spinBox_crossVal.minimum())
        self.ui.spinBox_crossVal.setDisabled(True)

        self.ui.spinBox_crossval_percent.setValue(self.ui.spinBox_crossval_percent.minimum())
        self.ui.spinBox_crossval_percent.setDisabled(True)

        # ACTIVATE ELEMENTS FOR SLIDING SPLIT
        self.ui.spinBox_sliding.setDisabled(False)
        self.ui.spinBox_sliding_percent.setDisabled(False)

    def on_next_val_click(self): 
        """
        Initiates the validation check workflow.
        """
        self._check_info_val()

    def _check_info_val(self):
        """
        Validates the user input and, if successful, proceeds to the model configuration stage.
        """
        if self._validate_data(): 
            self.ui.model_btn_2.setDisabled(False)
            self.ui.model_btn_1.setDisabled(False)
            self.ui.model_btn_2.setChecked(True)

    def _validate_data(self): 
        """
        Main validation logic for datasets and parameters.

        Returns
        -------
        bool
            True if validation passes, False otherwise.
        """
        if self.ui.radioButton_yes.isChecked():
            rows = self.ui.tableWidget_TVT.rowCount()
            columns = self.ui.tableWidget_TVT.columnCount()

            column_names = {}
            for col in range(columns):
                header_item = self.ui.tableWidget_TVT.horizontalHeaderItem(col)
                if header_item is not None:
                    column_names[header_item.text().strip().lower()] = col

            if "train" not in column_names or "test" not in column_names:
                logging.error("Las columnas 'train' y/o 'test' no existen en la tabla.")
            else:
                train_col = column_names["train"]
                val_col = column_names["validation"]
                test_col = column_names["test"]

                info_columns = {}

                for row in range(rows):
                    train_item = self.ui.tableWidget_TVT.item(row, train_col)
                    val_item = self.ui.tableWidget_TVT.item(row, val_col)
                    test_item = self.ui.tableWidget_TVT.item(row, test_col)

                    train_text = train_item.text().strip() if train_item else ""
                    val_text = val_item.text().strip() if val_item else ""
                    test_text = test_item.text().strip() if test_item else ""

                    if (train_text and not test_text) or (test_text and not train_text):
                        show_error(f"Error en la fila {row + 1}. Debes introducir contenido tanto en 'train' como en 'test'. La columna de 'validation' es opcional.")
                        return False
                
                    if train_text and test_text: 
                        info_columns[row] = {
                            "train": train_text, 
                            "val": val_text,
                            "test": test_text }
                        
                for row, datasets_dict in info_columns.items():
                    nombres_archivos = [datasets_dict['train'], datasets_dict['test']]
                    if datasets_dict['val']:
                        nombres_archivos.append(datasets_dict['val'])
                    datasets = self.datasetController.get_data()
                    datasets_seleccionados = [d for d in datasets.values() if d.name in nombres_archivos]

                    if len(datasets_seleccionados) != len(nombres_archivos):
                        show_error(f"Error en la fila {row + 1}. Uno o más datasets especificados no están disponibles.")
                        return False

                    ref_x_col = datasets_seleccionados[0].x_col
                    ref_y_col = datasets_seleccionados[0].y_col

                    for dataset in datasets_seleccionados[1:]:
                        if dataset.x_col != ref_x_col or dataset.y_col != ref_y_col:
                            show_error(f"Error en la fila {row + 1}. Los datasets deben tener las mismas columnas 'x' y la misma columna 'y'.")
                            return False
                
                self.datasetController.set_split_data(info_columns)

        if self.ui.radioButton_train_test.isChecked():
            if (self.ui.spinBox_train.value() == 0) | (self.ui.spinBox_test.value() == 0): 
                show_error("En la validación Train/Test debe rellenar los campos de 'Entrenamiento' y 'Testeo'. El campo opcional es 'Validación'.")
                return False
            else: 
                if self.ui.spinBox_train.value() + self.ui.spinBox_test.value() + self.ui.spinBox_val.value() != 100.0:
                    show_error("Los porcentajes deben sumar 100.0")
                    return False
                else: 
                    self.validation_config = {
                        "type": "train/test", 
                        "train": self.ui.spinBox_train.value(), 
                        "validation": self.ui.spinBox_val.value(), 
                        "test": self.ui.spinBox_test.value()
                    }
                    self.validationController.set_validation_config(self.validation_config)
                    return True

        elif self.ui.radioButton_crossVal.isChecked():
            if (self.ui.spinBox_crossVal.value() == 0) or (self.ui.spinBox_crossval_percent.value() == 0):
                show_error("Debe rellenar el campo para crossValidation y el porcentaje de test.")
                return False
            else: 
                self.validation_config = {
                        "type": "crossVal", 
                        "crossVal": self.ui.spinBox_crossVal.value(),
                        "percentage_crossVal": self.ui.spinBox_crossval_percent.value()
                }
                self.validationController.set_validation_config(self.validation_config)
                return True
            
        elif self.ui.radioButton_sliding.isChecked():
            if (self.ui.spinBox_sliding.value() == 0) or (self.ui.spinBox_sliding_percent.value() == 0):
                show_error("Debe rellenar el campo para Sliding Split y el porcentaje de test.")
                return False
            else: 
                self.validation_config = {
                        "type": "Sliding_split", 
                        "sliding": self.ui.spinBox_sliding.value(), 
                        "percentage_sliding": self.ui.spinBox_sliding_percent.value()
                }
                self.validationController.set_validation_config(self.validation_config)
                return True

    def on_radioButton_no_click(self):
        """
        Disables manual assignment widgets when 'No' is selected.
        """
        self.ui.listWidget_datasets.setDisabled(True)
        self.ui.tableWidget_TVT.setDisabled(True)

    def on_radioButton_yes_click(self):
        """
        Enables manual assignment widgets when 'Yes' is selected.
        """
        self.ui.listWidget_datasets.setDisabled(False)
        self.ui.tableWidget_TVT.setDisabled(False)