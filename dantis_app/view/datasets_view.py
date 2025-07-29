from PyQt5.QtWidgets import QWidget, QDialog

from view.dialogs.dialogAddDataset import DialogAddDataset
from controller.dataset_controller import DatasetController
from core.utils import show_error
from itertools import combinations

class ViewDatasets(QWidget):
    """
    GUI component for managing datasets within the application.  
    This class handles user interaction for adding, validating, and listing datasets 
    using a graphical interface built with PyQt5. It interacts with the `DatasetController` 
    for backend dataset logic and integrates UI elements for navigation and validation.

    Parameters
    ----------
    ui : object
        A reference to the main UI object that contains the visual components.

    Attributes
    ----------
    ui : object
        The user interface reference.
    global_dataset_id : int
        A global identifier for dataset tracking.
    datasetController : DatasetController
        Controller instance for dataset logic and data management.

    Methods
    -------
    init_ui()
        Initializes the user interface bindings for buttons and interaction.
    on_add_dataset_click()
        Opens a dialog for adding a new dataset.
    on_next_dataset_click()
        Proceeds to the next dataset screen after validation.
    _check_info_dataset()
        Validates that datasets are correctly populated and unique.
    _validate_data()
        Validates each dataset for proper feature selection and duplication.
    _check_ListAndTabla()
        Updates the UI list and table to reflect current datasets.
    """

    def __init__(self, ui):
        """
        Initialize the ViewDatasets component.

        Parameters
        ----------
        ui : object
            The UI object with necessary widgets and controls.
        """
        super().__init__()
        self.ui = ui
        self.global_dataset_id = 1
        self.datasetController = DatasetController()
        self.init_ui()

    def init_ui(self):
        """
        Sets up UI interactions such as button clicks for adding and validating datasets.
        """
        self.ui.datasets_btn_2.setChecked(True)
        self.ui.add_dataset.clicked.connect(self.on_add_dataset_click)
        self.ui.next_dataset.clicked.connect(self.on_next_dataset_click)

    def on_add_dataset_click(self):
        """
        Triggered when the user clicks the 'Add Dataset' button.
        Opens a dialog to add a new dataset using `DialogAddDataset`.
        """
        dialog = DialogAddDataset(self.ui, parent=self , datasetController=self.datasetController)
        dialog.exec() == QDialog.Accepted

    def on_next_dataset_click(self): 
        """
        Triggered when the user clicks the 'Next' button.
        Validates dataset inputs before proceeding to the next step.
        """    
        self._check_info_dataset()

    def _check_info_dataset(self):
        """
        Validates datasets for correct configuration.
        Enables buttons and navigation if validation passes.
        """
        if self._validate_data(): 
            self.ui.val_btn_2.setDisabled(False)
            self.ui.val_btn_1.setDisabled(False)
            self.ui.val_btn_2.setChecked(True)

    def _validate_data(self): 
        """
        Validates the dataset configurations:
        - Ensures each dataset has at least one selected feature (`x_col`)
        - Checks for duplicated datasets (by name)
        - Displays appropriate error messages if validation fails

        Returns
        -------
        bool
            True if all datasets are valid and unique, False otherwise.
        """
        content = self.ui.scrollArea_3.widget()
        data = self.datasetController.get_data()
        checkboxes_empty = False
        if content is not None and content.findChildren(QWidget):
            for dataset_id, dataset_info in data.items():
                if not any(dataset_info.x_col.values()):
                    checkboxes_empty = True
                    break

            if checkboxes_empty:
                show_error("Para poder continuar, debe seleccionar al menos una opción de x_col en cada dataset.")
                return False          

            def _equal_datasets(d1, d2):
                if d1.name == d2.name:
                    return True
                return False

            ids = list(data.keys())
            for id1, id2 in combinations(ids, 2):
                if _equal_datasets(data[id1], data[id2]):
                    show_error(f"¡Atención! Los datasets {id1} y {id2} son tienen el mismo nombre. Por favor elimina uno de ellos.")
                    return False

            self._check_ListAndTabla()
            return True
        
        else:
            show_error("Para poder continuar, debe agregar algún dataset para que los modelos se puedan entrenar.")
            return False

    def _check_ListAndTabla(self):
        """
        Synchronizes the list widget and table view with the current datasets:
        - Adds new dataset names to the list if missing
        - Removes deleted datasets from the list
        - Ensures the table content also reflects current datasets
        """
        data = self.datasetController.get_data()
        names = [dataset.name for dataset in data.values()]

        existing_list_items = [self.ui.listWidget_datasets.item(i).text() 
                               for i in range(self.ui.listWidget_datasets.count())
        ]

        existing_table_items = {}
        for row in range(self.ui.tableWidget_TVT.rowCount()):
            for col in range(self.ui.tableWidget_TVT.columnCount()):
                item = self.ui.tableWidget_TVT.item(row, col)
                if item is not None:
                    existing_table_items[(row, col)] = item.text()

        excluded_items = set(existing_list_items + list(existing_table_items.values()))

        for name in names:
            if name not in excluded_items:
                self.ui.listWidget_datasets.addItem(name)

        for i in reversed(range(self.ui.listWidget_datasets.count())):
            item_text = self.ui.listWidget_datasets.item(i).text()
            if item_text not in names:
                self.ui.listWidget_datasets.takeItem(i)

        for (row, col), text in existing_table_items.items():
            if text not in names:
                self.ui.tableWidget_TVT.setItem(row, col, None)