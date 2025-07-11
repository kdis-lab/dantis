from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QDialog, QFileDialog, QSizePolicy
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QHBoxLayout, QWidget, QLineEdit
import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics

class DatasetItem(QWidget):
    def __init__(self, id, name):
        super().__init__()
        self.id = id
        self.name = name
        self.full_text = f"{name} (ID {id})"

        layout = QHBoxLayout()

        # QLabel que crece según espacio
        self.label = QLabel(self.full_text)
        self.label.setToolTip(self.full_text)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.label.setMinimumWidth(30)  # mínimo para no desaparecer
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Botón "Cargar"
        self.button = QPushButton("Cargar")
        self.button.clicked.connect(self._select_file)

        # Campo para mostrar path
        self.path_line = QLineEdit()
        self.path_line.setReadOnly(True)
        self.path_line.setStyleSheet("font-size: 10px; color: gray")
        self.path_line.setFixedWidth(150)
        self.path_line.textChanged.connect(self._on_path_changed)

        # Añadir al layout
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.path_line)

        self.setLayout(layout)

    def _select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar modelo", "", "Modelos (*.joblib *.pkl)"
        )
        if file_path:
            self.path_line.setText(file_path)

    def _on_path_changed(self):
        # Llama al método del diálogo (si lo conoces)
        parent = self.parent()
        while parent and not isinstance(parent, DialogLoadModel):
            parent = parent.parent()
        if parent:
            parent.check_ready()

class DialogLoadModel(QDialog):
    def __init__(self, parent=None, modelController=None, datasetController=None, id=None, preloaded_paths=None):
        super().__init__(parent)
        self.setWindowTitle("Cargar modelos .joblib entrenados")
        self.parent_window = parent
        self.modelController = modelController
        self.id = id
        self.model_path = None
        self.selected_dataset_id = None
        self.datasetController = datasetController

        self.preloaded_paths = preloaded_paths # <<< PATHS PREVIOS
        self.dataset_paths = {}  # <<< NUEVOS PATHS SE GUARDARÁN AQUÍ

        self.init_ui()

        self.setMinimumSize(600, 400)

    def init_ui(self):
        layout = QVBoxLayout()

        self.dataset_label = QLabel("Seleccione el .joblib del modelo entrenado para el datasets que desee cargar:")
        layout.addWidget(self.dataset_label)

        self.dataset_list = QListWidget()
        self.dataset_list_split = QListWidget()
        layout.addWidget(self.dataset_list)
        layout.addWidget(self.dataset_list_split)

        self.btn_save = QPushButton("Guardar modelo cargado")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_model)
        layout.addWidget(self.btn_save)

        self.setLayout(layout)

        # Llenar la lista de datasets
        self.list_datasets()

    def list_datasets(self):
        datasets, split_datasets = self.datasetController.get_all_datasets_names()

        # Datasets completos
        header_completos = QListWidgetItem("📂 Datasets Completos")
        header_completos.setFlags(Qt.NoItemFlags)
        self.dataset_list.addItem(header_completos)

        for ds_id, name in datasets.items():
            item_widget = DatasetItem(ds_id, name)
            if ds_id in self.preloaded_paths["full"]:
                item_widget.path_line.setText(self.preloaded_paths["full"][ds_id])
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.dataset_list.addItem(list_item)
            self.dataset_list.setItemWidget(list_item, item_widget)

        # Datasets particionados
        header_split = QListWidgetItem("\n📦 Grupos particionados")
        header_split.setFlags(Qt.NoItemFlags)
        self.dataset_list_split.addItem(header_split)

        for group_id, names in split_datasets.items():
            nombres_str = ", ".join(n for n in names if n)
            item_widget = DatasetItem(group_id, nombres_str)
            if group_id in self.preloaded_paths["split"]:
                item_widget.path_line.setText(self.preloaded_paths["split"][group_id])
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.dataset_list_split.addItem(list_item)
            self.dataset_list_split.setItemWidget(list_item, item_widget)

    def on_dataset_selected(self, current, previous):
        if current:
            self.selected_dataset_id = current.data(1)
            self.check_ready()


    def check_ready(self):
        def any_loaded(dataset_list):
            return any(
                isinstance(dataset_list.itemWidget(dataset_list.item(i)), DatasetItem) and 
                dataset_list.itemWidget(dataset_list.item(i)).path_line.text().strip()
                for i in range(dataset_list.count())
            )

        self.btn_save.setEnabled(
            any_loaded(self.dataset_list) or any_loaded(self.dataset_list_split)
        )


    def save_list(self, list_widget, key, dataset_paths):
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            widget = list_widget.itemWidget(item)
            if isinstance(widget, DatasetItem):
                path = widget.path_line.text().strip()
                if path:
                    dataset_paths[key][widget.id] = path

    def save_model(self):
        dataset_paths = {"full": {}, "split": {}}
        self.save_list(self.dataset_list, "full", dataset_paths)
        self.save_list(self.dataset_list_split, "split", dataset_paths)

        self.dataset_paths = dataset_paths
        self.accept()
        
