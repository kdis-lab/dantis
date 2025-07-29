from PyQt5.QtWidgets import QDialog, QVBoxLayout, QRadioButton, QHBoxLayout, QCheckBox
from PyQt5.QtWidgets import QFormLayout, QGroupBox, QWidget, QFileDialog
from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton, QSizePolicy
from PyQt5.QtWidgets import QLineEdit, QScrollArea
from PyQt5.QtCore import Qt

from controller.dataset_controller import load_predefined_datasets_name
import pandas as pd
from view.dialogs.dialogDataVisualizer import DatasetVisualizer
import os

class DialogAddDataset(QDialog):
    def __init__(self, ui, parent=None, datasetController = None):
        """
        Initialize the dataset dialog.

        Parameters
        ----------
        ui : QWidget
            The main user interface.
        parent : QWidget, optional
            The parent window.
        datasetController : DatasetController
            Controller for managing dataset logic.
        """
        super().__init__(parent)     
        self.parent_window = parent
        self.stacked_widget = self.parent_window.ui.stackedWidget
        self.ui = ui
        self.datasetController = datasetController
        self.datasets_widgets = []
        self.github_repo_url = "https://api.github.com/repos/Elenitaalva/Datasets/contents"
        self.datasets_predefinidos = load_predefined_datasets_name(self.github_repo_url)
        self.setWindowTitle("Añadir Dataset")        
        self.init_widgets()
        self.init_layout()
        self.init_connections()
        self.setLayout(self.main_layout)

    def init_widgets(self):
        """
        Initialize all UI widgets including radio buttons,
        labels, buttons, and input fields.
        """
        self.radio_predefined = QRadioButton("Dataset predefinido")
        self.radio_local = QRadioButton("Local")
        self.radio_url = QRadioButton("URL")
        self.radio_predefined.setChecked(True)

        self.predefined_label = QLabel("\nSelecciona un dataset predefinido:")
        self.predefined_button = QComboBox()
        self.predefined_button.addItems(self.datasets_predefinidos)
        self.predefined_button_load = QPushButton("Cargar Dataset predefinido")

        self.local_label = QLabel("\nSelecciona un dataset de forma local: ")
        self.local_button = QPushButton("Seleccionar archivo")

        self.url_label = QLabel("Introduce la URL del dataset:")
        self.url_input = QLineEdit()
        self.url_button = QPushButton("Cargar Dataset desde URL")

        self.preview_label = QLabel("\nArchivo cargado:")
        self.preview_text = QLabel("")

        self.accept_button = QPushButton("Aceptar")
        self.cancel_button = QPushButton("Cancelar")

    def init_layout(self):
        """
        Set up the layout for the dialog including form and button placement.
        """
        self.main_layout = QVBoxLayout()

        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_predefined)
        radio_layout.addWidget(self.radio_local)
        radio_layout.addWidget(self.radio_url)

        self.form_layout = QFormLayout()
        self.form_layout.addRow(self.predefined_label)
        self.form_layout.addRow(self.predefined_button)
        self.form_layout.addRow(self.predefined_button_load)
        self.form_layout.addRow(self.local_label)
        self.form_layout.addRow(self.local_button)
        self.form_layout.addRow(QLabel(" "))
        self.form_layout.addRow(self.url_label, self.url_input)
        self.form_layout.addRow(self.url_button)

        self.main_layout.addLayout(radio_layout)
        self.main_layout.addLayout(self.form_layout)
        self.main_layout.addWidget(self.preview_label)
        self.main_layout.addWidget(self.preview_text)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.accept_button)
        button_layout.addWidget(self.cancel_button)
        self.main_layout.addLayout(button_layout)

    def init_connections(self):
        """Connects signals to corresponding slot methods."""
        self.radio_predefined.toggled.connect(self.toggle_input_method)
        self.radio_local.toggled.connect(self.toggle_input_method)
        self.radio_url.toggled.connect(self.toggle_input_method)

        self.predefined_button_load.clicked.connect(self.load_github_dataset)
        self.local_button.clicked.connect(self.load_local_file)
        self.url_button.clicked.connect(self.load_url_file)

        self.accept_button.clicked.connect(self.confirm_selection)
        self.cancel_button.clicked.connect(self.reject)

    def toggle_input_method(self):
        """
        Toggle dataset input method based on selected radio button.
        Enables/disables widgets accordingly.
        """
        if self.radio_predefined.isChecked():
            self.predefined_button.setEnabled(True)
            self.predefined_button_load.setEnabled(True)
            self.local_button.setEnabled(False)
            self.url_input.setDisabled(True)
            self.url_button.setDisabled(True)
            self.preview_text.setText("")
        elif self.radio_local.isChecked():
            # If "Local" is selected, enable the file selection button
            self.predefined_button.setEnabled(False)
            self.predefined_button_load.setEnabled(False)
            self.local_button.setEnabled(True)
            self.url_input.setDisabled(True)
            self.url_button.setDisabled(True)
            self.preview_text.setText("") 
        elif self.radio_url.isChecked():
            # If "URL" is selected, enable the text field and button
            self.predefined_button.setEnabled(False)
            self.predefined_button_load.setEnabled(False)
            self.local_button.setEnabled(False)
            self.url_input.setEnabled(True)
            self.url_button.setEnabled(True)
            self.preview_text.setText("") 
        
    def load_local_file(self):
        """
        Open file dialog to select local CSV files and show preview.
        """
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, "Seleccionar archivos", "", "Archivos CSV (*.csv);;Todos los archivos (*)")
        
        if file_paths:
            self.file_path = file_paths
            file_names = [os.path.basename(path) for path in file_paths]
            self.preview_text.setText("\n".join(file_names))

    def load_github_dataset(self):   
        """
        Load a predefined dataset from GitHub.
        """
        value = self.predefined_button.currentText()
        if value:
            self.file_path = f"https://raw.githubusercontent.com/Elenitaalva/Datasets/main/{value}"
            self.preview_text.setText(value)

    def load_url_file(self): 
        """
        Load dataset from a specified URL.
        """
        value = self.url_input.text()
        self.file_path = value
        self.preview_text.setText(value)

    def confirm_selection(self):
        """
        Confirm user selection and add dataset to controller.
        """
        if self.radio_local.isChecked() and self.preview_text.text().strip():
            for file_path in self.file_path:
                id = self.datasetController.add_datasets(file_path, 0)
                self.create_dataset_layout(id)

        elif self.radio_predefined.isChecked() and self.preview_text.text().strip():
            id = self.datasetController.add_datasets(self.file_path, 1)
            self.create_dataset_layout(id)         

        elif self.radio_url.isChecked() and self.preview_text.text().strip():
            id = self.datasetController.add_datasets(self.file_path, 2)
            self.create_dataset_layout(id) 

        self.accept()
            
    def update_y_col(self):
        """
        Update available x column checkboxes when y column changes.
        """
        new_groups = []
        for grupo in self.datasets_widgets:
            x_checkboxes = grupo.get("x_checkboxes")
            y_combo = grupo.get("y_combo")
            if y_combo is None or y_combo.parent() is None:
                continue  
            y_col = y_combo.currentText()
            for cb in x_checkboxes:
                if cb.text() == y_col:
                    cb.setChecked(False)
                    cb.setEnabled(False)
                else:
                    cb.setEnabled(True)
            new_groups.append(grupo)
        self.datasets_widgets = new_groups

    def get_url(self):
        """
        Get the current text from the URL input field.

        Returns
        -------
        str
            URL string entered by the user.
        """
        return self.url_input.text()

    def handle_x_col_change(self, id, col_name, state, col_check_states):
        """
        Handle x column checkbox state change.

        Parameters
        ----------
        id : int
            Dataset ID.
        col_name : str
            Column name.
        state : int
            Qt.Checked or Qt.Unchecked.
        col_check_states : dict
            Current checkbox states.
        """
        col_check_states[col_name] = (state == Qt.Checked)
        self.datasetController.set_options_x_col(col_check_states)
        self.datasetController.add_x_col(id, col_check_states)

    def handle_y_col_change(self, id, y_col_combo, selected_y):
        """
        Handle y column change in combo box.

        Parameters
        ----------
        id : int
            Dataset ID.
        y_col_combo : QComboBox
            Combo box widget.
        selected_y : str
            Selected y column name.
        """
        selected_y = y_col_combo.currentText()
        self.datasetController.add_y_col(id, selected_y)

    def visualization_dataset(self, id):
        """
        Open a dataset visualizer window for the selected dataset.

        Parameters
        ----------
        id : int
            Dataset ID to visualize.
        """
        dataset = self.datasetController.data[id].data
        if isinstance(dataset, pd.DataFrame):
            visualizer = DatasetVisualizer(dataset)
            visualizer.show()

    def clear_scrollArea3(self):
        """
        Clear all widgets from scrollArea_3.
        """
        scroll_area = self.stacked_widget.widget(0).findChild(QScrollArea, "scrollArea_3")
        scroll_content = scroll_area.widget()
        if scroll_content is None:
            print("El scrollArea no tiene contenido.")
            return

        layout = scroll_content.layout()
        if layout is None:
            print("El contenido del scrollArea no tiene layout.")
            return

        while layout.count():
            item = layout.takeAt(0)
            if item.layout():
                sublayout = item.layout()
                while sublayout.count():
                    subitem = sublayout.takeAt(0)
                    widget = subitem.widget()
                    if widget:
                        widget.setParent(None)
            else:
                widget = item.widget()
                if widget:
                    widget.setParent(None)

        self.datasets_widgets.clear()
    
    def aniadir_nuevos_datos(self):
        """
        Add new datasets to scrollArea layout.
        """
        self.clear_scrollArea3()
        # Recorremos todos los datasets actuales en el controller
        for id in self.datasetController.get_data():
            self.create_dataset_layout(id)

    def delete_dataset(self, dataset_id, container_widget):
        """
        Delete a dataset and remove its UI representation.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset to delete.
        container_widget : QWidget
            Widget to remove from layout.
        """
        if self.datasetController.delete(dataset_id):
            container_widget.setParent(None)

            # Limpiar datasets_widgets eliminando el que corresponde
            self.datasets_widgets = [
                grupo for grupo in self.datasets_widgets
                if self.datasetController.get_y_col(dataset_id) != grupo["y_combo"].currentText()
            ]

            self.clear_scrollArea3()
            self.aniadir_nuevos_datos()
        else:
            print("Error al eliminar dataset con ID:", dataset_id)
    

    def _add_to_scroll_area(self, widget):
        """
        Add a widget to the scroll area.

        Parameters
        ----------
        widget : QWidget
            Widget to add.
        """
        scroll_area = self.stacked_widget.widget(0).findChild(QScrollArea, "scrollArea_3")
        scroll_content = scroll_area.widget()

        if scroll_content.layout() is None:
            scroll_content.setLayout(QVBoxLayout())
        scroll_content.layout().setAlignment(Qt.AlignTop)
        scroll_content.layout().addWidget(widget)

    def _create_x_col_group(self, dataset_id, columnas):
        """
        Create a scrollable group of x column checkboxes.

        Parameters
        ----------
        dataset_id : int
            Dataset ID.
        columnas : list
            List of column names.

        Returns
        -------
        QGroupBox
            Group box containing x column checkboxes.
        list
            List of QCheckBox widgets.
        """
        x_col_values = self.datasetController.get_options_x_col(dataset_id) or {}
        checkbox_layout = QVBoxLayout()
        x_col_checkboxes = []
        col_check_states = {}

        for col in columnas:
            checkbox = QCheckBox(col)
            is_checked = x_col_values.get(col, True)
            checkbox.setChecked(is_checked)
            col_check_states[col] = is_checked

            checkbox.stateChanged.connect(
                lambda state, c=col: self.handle_x_col_change(dataset_id, c, state, col_check_states)
            )
            checkbox_layout.addWidget(checkbox)
            x_col_checkboxes.append(checkbox)

        checkbox_layout.addStretch()

        checkbox_widget = QWidget()
        checkbox_widget.setLayout(checkbox_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(checkbox_widget)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        scroll_area.setFixedHeight(100)
        scroll_area.setFixedWidth(250)

        x_col_layout = QVBoxLayout()
        x_col_layout.addWidget(scroll_area)

        x_col_group = QGroupBox("Características")
        x_col_group.setLayout(x_col_layout)

        self.datasetController.options_x_col = {cb.text(): cb.isChecked() for cb in x_col_checkboxes}
        self.datasetController.add_x_col(dataset_id, self.datasetController.options_x_col)

        return x_col_group, x_col_checkboxes

    def _create_y_col_selector(self, dataset_id, columnas):
        """
        Create y column selection dropdown.

        Parameters
        ----------
        dataset_id : int
            Dataset ID.
        columnas : list
            List of column names.

        Returns
        -------
        QLabel
            Label for y column.
        QComboBox
            Dropdown for selecting y column.
        """
        y_col_value = self.datasetController.get_y_col(dataset_id)
        y_col_combo = QComboBox()
        y_col_combo.addItems(columnas)
        y_label = QLabel("y_col")

        if y_col_value in columnas:
            y_col_combo.setCurrentText(y_col_value)
        else:
            y_col_combo.setCurrentIndex(len(columnas) - 1)
            y_col_value = y_col_combo.currentText()

        self.datasetController.add_y_col(dataset_id, y_col_value)

        def on_y_col_changed():
            selected_y = y_col_combo.currentText()
            self.handle_y_col_change(dataset_id, y_col_combo, selected_y)
            self.update_y_col()

        y_col_combo.currentIndexChanged.connect(on_y_col_changed)

        return y_label, y_col_combo

    def _create_visualization_button(self, dataset_id):
        """
        Create a visualization button for the dataset.

        Parameters
        ----------
        dataset_id : int
            Dataset ID.

        Returns
        -------
        QPushButton
            Visualization button.
        """
        button = QPushButton("Visualización")
        button.clicked.connect(lambda: self.visualization_dataset(dataset_id))
        return button

    def create_dataset_layout(self, dataset_id):
        """
        Create and display the layout for a loaded dataset, including
        x column checkboxes, y column selector, visualization and delete buttons.

        Parameters
        ----------
        dataset_id : int
            Dataset ID.
        """
        file_path = self.datasetController.get_data().get(dataset_id)
        columnas = self.datasetController.get_columns(dataset_id)
        loaded_file_label = QLabel(f"Dataset {dataset_id}: {file_path.name}")
       
        x_col_group, x_col_checkboxes = self._create_x_col_group(dataset_id, columnas)
        y_label, y_col_combo = self._create_y_col_selector(dataset_id, columnas)
        visualization_button = self._create_visualization_button(dataset_id)
        delete_button = QPushButton("Eliminar")

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(x_col_group)
        controls_layout.addWidget(y_label)
        controls_layout.addWidget(y_col_combo)
        controls_layout.addWidget(visualization_button)
        controls_layout.addWidget(delete_button)

        dataset_container_layout = QVBoxLayout()
        dataset_container_layout.addWidget(loaded_file_label)
        dataset_container_layout.addLayout(controls_layout)

        dataset_container_widget = QWidget()
        dataset_container_widget.setLayout(dataset_container_layout)

        self._add_to_scroll_area(dataset_container_widget)

        delete_button.clicked.connect(lambda: self.delete_dataset(dataset_id, dataset_container_widget))

        self.datasets_widgets.append({
            "x_checkboxes": x_col_checkboxes,
            "y_label": y_label,
            "y_combo": y_col_combo
        })

        self.update_y_col()