"""

This module provides dialog components to load pretrained model files (.joblib or .pkl)
and associate them with datasets used in the application.

Classes
-------
DatasetItem : QWidget
    Custom widget representing a single dataset entry with a file selector.
DialogLoadModel : QDialog
    Dialog window that allows the user to associate trained models with datasets.
"""

from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QDialog, QFileDialog, QSizePolicy
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QHBoxLayout, QWidget, QLineEdit
from PyQt5.QtCore import Qt

class DatasetItem(QWidget):
    """
    A custom widget representing a dataset entry with a label, file selector button,
    and a read-only path field to load a model.

    Parameters
    ----------
    id : int or str
        The unique identifier for the dataset.
    name : str
        The human-readable name of the dataset.

    Attributes
    ----------
    id : int or str
        Dataset identifier.
    name : str
        Dataset name.
    full_text : str
        Combined label text including the name and ID.
    label : QLabel
        Displays the dataset name and ID.
    button : QPushButton
        Button to open a file dialog for model selection.
    path_line : QLineEdit
        Read-only field that displays the selected model file path.

    Methods
    -------
    _select_file()
        Opens a QFileDialog to choose a model file.
    _on_path_changed()
        Triggers the parent dialog's readiness check when a file path is set.
    """
    def __init__(self, id, name):
        super().__init__()
        self.id = id
        self.name = name
        self.full_text = f"{name} (ID {id})"

        layout = QHBoxLayout()
        self.label = QLabel(self.full_text)
        self.label.setToolTip(self.full_text)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.label.setMinimumWidth(30) 
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.button = QPushButton("Cargar")
        self.button.clicked.connect(self._select_file)

        self.path_line = QLineEdit()
        self.path_line.setReadOnly(True)
        self.path_line.setStyleSheet("font-size: 10px; color: gray")
        self.path_line.setFixedWidth(150)
        self.path_line.textChanged.connect(self._on_path_changed)

        layout.addWidget(self.label)
        layout.addWidget(self.button)
        layout.addWidget(self.path_line)

        self.setLayout(layout)

    def _select_file(self):
        """
        Opens a file dialog to select a model file and sets its path in the UI.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar modelo", "", "Modelos (*.joblib *.pkl)"
        )
        if file_path:
            self.path_line.setText(file_path)

    def _on_path_changed(self):
        """
        Notifies the parent dialog when the model path changes,
        triggering a readiness check.
        """
        parent = self.parent()
        while parent and not isinstance(parent, DialogLoadModel):
            parent = parent.parent()
        if parent:
            parent.check_ready()

class DialogLoadModel(QDialog):
    """
    Dialog to associate pretrained model files with full or split datasets.

    Parameters
    ----------
    parent : QWidget, optional
        Parent window (default is None).
    modelController : object, optional
        Controller used to manage model associations.
    datasetController : object, optional
        Controller used to access datasets.
    id : str or int, optional
        Optional identifier for internal tracking.
    preloaded_paths : dict, optional
        Dictionary containing already loaded model paths for datasets.

    Attributes
    ----------
    dataset_list : QListWidget
        Widget that displays full datasets.
    dataset_list_split : QListWidget
        Widget that displays split/grouped datasets.
    btn_save : QPushButton
        Button to confirm and save loaded models.
    dataset_paths : dict
        Dictionary storing new model paths selected by the user.

    Methods
    -------
    init_ui()
        Builds and lays out the UI components.
    list_datasets()
        Populates the dataset widgets with current datasets and models.
    on_dataset_selected(current, previous)
        Triggered when a dataset item is selected.
    check_ready()
        Enables the save button if any models have been assigned.
    save_list(list_widget, key, dataset_paths)
        Extracts selected paths from list widgets and stores them.
    save_model()
        Finalizes selections and closes the dialog.
    """
    def __init__(self, parent=None, modelController=None, datasetController=None, id=None, preloaded_paths=None):
        """
        Initializes the DialogLoadModel window for assigning trained model files to datasets.

        This method sets up the main dialog window, loads any pre-associated model paths,
        and prepares the UI layout to allow model assignment.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget or window. Defaults to None.
        modelController : object, optional
            The controller responsible for managing model logic. Defaults to None.
        datasetController : object, optional
            Controller that handles the retrieval of dataset metadata. Defaults to None.
        id : int or str, optional
            Identifier that may be used for context tracking. Defaults to None.
        preloaded_paths : dict, optional
            Dictionary with keys `"full"` and `"split"` mapping dataset IDs to preloaded model paths.
            If not provided, defaults to empty path dictionaries.
        
        Attributes
        ----------
        parent_window : QWidget
            Reference to the parent window.
        modelController : object
            Logic controller for handling model-related operations.
        datasetController : object
            Controller used to retrieve dataset information.
        id : int or str
            Optional context-specific identifier.
        preloaded_paths : dict
            Dictionary of preloaded model paths by dataset type (`"full"` or `"split"`).
        dataset_paths : dict
            Will store the paths selected by the user during dialog interaction.
        dataset_list : QListWidget
            List widget displaying full datasets.
        dataset_list_split : QListWidget
            List widget displaying split (grouped) datasets.
        btn_save : QPushButton
            Button that saves the model selections; enabled only when at least one model is selected.
        """
        super().__init__(parent)
        self.setWindowTitle("Cargar modelos .joblib entrenados")
        self.parent_window = parent
        self.modelController = modelController
        self.id = id
        self.model_path = None
        self.selected_dataset_id = None
        self.datasetController = datasetController

        self.preloaded_paths = preloaded_paths 
        self.dataset_paths = {}

        self.init_ui()
        self.setMinimumSize(600, 400)

    def init_ui(self):
        """
        Initializes and lays out all widgets in the dialog.
        """
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
        self.list_datasets()

    def list_datasets(self):
        """
        Populates the full and split dataset lists with custom `DatasetItem` widgets.
        If paths were preloaded, restores them in their respective fields.
        """
        datasets, split_datasets = self.datasetController.get_all_datasets_names()

        # Full dataset
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

        # Split datasets
        header_split = QListWidgetItem("\n📦 Grupos particionados")
        header_split.setFlags(Qt.NoItemFlags)
        self.dataset_list_split.addItem(header_split)

        for group_id, names in split_datasets.items():
            names_str = ", ".join(n for n in names if n)
            item_widget = DatasetItem(group_id, names_str)
            if group_id in self.preloaded_paths["split"]:
                item_widget.path_line.setText(self.preloaded_paths["split"][group_id])
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.dataset_list_split.addItem(list_item)
            self.dataset_list_split.setItemWidget(list_item, item_widget)

    def on_dataset_selected(self, current):
        """
        Triggered when a dataset in the list is selected.

        Parameters
        ----------
        current : QListWidgetItem
            The currently selected item.
        """
        if current:
            self.selected_dataset_id = current.data(1)
            self.check_ready()

    def check_ready(self):
        """
        Checks whether at least one model has been selected in either dataset list.
        Enables or disables the save button accordingly.
        """
        def any_loaded(dataset_list):
            """
            Checks if any `DatasetItem` in a given list has a model file path selected.

            This is a helper function used in `check_ready()` to determine whether
            the save button (`btn_save`) should be enabled, depending on whether
            at least one dataset has a valid model path assigned.

            Parameters
            ----------
            dataset_list : QListWidget
                The list widget containing `DatasetItem` entries to check.

            Returns
            -------
            bool
                True if at least one `DatasetItem` in the list has a non-empty path;
                otherwise, False.
            """
            return any(
                isinstance(dataset_list.itemWidget(dataset_list.item(i)), DatasetItem) and 
                dataset_list.itemWidget(dataset_list.item(i)).path_line.text().strip()
                for i in range(dataset_list.count())
            )

        self.btn_save.setEnabled(
            any_loaded(self.dataset_list) or any_loaded(self.dataset_list_split)
        )


    def save_list(self, list_widget, key, dataset_paths):
        """
        Gathers all model paths selected by the user in the provided list.

        Parameters
        ----------
        list_widget : QListWidget
            The widget containing dataset items.
        key : str
            Whether it's a 'full' or 'split' dataset group.
        dataset_paths : dict
            The dictionary in which to store selected paths.
        """
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            widget = list_widget.itemWidget(item)
            if isinstance(widget, DatasetItem):
                path = widget.path_line.text().strip()
                if path:
                    dataset_paths[key][widget.id] = path

    def save_model(self):
        """
        Finalizes model-path selections by collecting all file paths and storing them
        in `self.dataset_paths`. Then closes the dialog with accept().
        """
        dataset_paths = {"full": {}, "split": {}}
        self.save_list(self.dataset_list, "full", dataset_paths)
        self.save_list(self.dataset_list_split, "split", dataset_paths)

        self.dataset_paths = dataset_paths
        self.accept()
        
