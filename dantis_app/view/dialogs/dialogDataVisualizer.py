from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QLabel 
from PyQt5.QtWidgets import QPushButton, QMenu, QMainWindow, QAction
from PyQt5.QtWidgets import QLineEdit, QTableWidget, QTableWidgetItem
from matplotlib.figure import Figure
from PyQt5.QtCore import QPoint
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import pandas as pd
from matplotlib.dates import DateFormatter

class DatasetVisualizer(QMainWindow):
    """
    Main window for dataset visualization including table and plot with filtering and sorting.

    Attributes
    ----------
    dataset : pandas.DataFrame
        The dataset to be visualized.
    plot_canvas : PlotCanvas
        The matplotlib canvas for plotting data.
    table_widget : QTableWidget
        Widget showing the dataset in tabular form.
    column_selector : QComboBox
        Dropdown to select the column to filter by.
    filter_input : QLineEdit
        Text input for filter values.
    filter_button : QPushButton
        Button to apply the filter.
    clear_filter_button : QPushButton
        Button to clear the filter and reset views.

    Methods
    -------
    update_table(data, table_widget)
        Populate the table widget with the given DataFrame.
    apply_filter(column_selector, filter_input, plot_canvas, table_widget)
        Apply filter to the dataset and update the table and plot.
    clear_filter(filter_input, table_widget, plot_canvas, filter_button, column_selector)
        Clear the current filter and restore the original dataset view.
    show_sort_menu(column_index, plot_canvas, table_widget)
        Display sorting options menu for the table column header.
    """
    def __init__(self, dataset, parent=None):
        """
        Initialize the DatasetVisualizer window.

        Parameters
        ----------
        dataset : pandas.DataFrame
            The dataset to visualize.
        parent : QWidget, optional
            Parent widget.
        """
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle("Visualización del Dataset")
        self.setGeometry(200, 200, 1000, 800)

        self.plot_canvas = PlotCanvas()
        self.plot_canvas.plot(self.dataset)
        toolbar = NavigationToolbar(self.plot_canvas, self)

        self.table_widget = QTableWidget()
        self.update_table(self.dataset, self.table_widget)

        self.column_selector = QComboBox()
        self.column_selector.addItems(self.dataset.columns)

        self.filter_input = QLineEdit()

        self.filter_button = QPushButton("Aplicar Filtro")
        self.clear_filter_button = QPushButton("Limpiar Filtro")

        self.filter_button.clicked.connect(lambda: self.apply_filter(
            self.column_selector, self.filter_input, self.plot_canvas, self.table_widget
        ))
        self.clear_filter_button.clicked.connect(lambda: self.clear_filter(
            self.filter_input, self.table_widget, self.plot_canvas,
            self.filter_button, self.column_selector
        ))

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Columna:"))
        filter_layout.addWidget(self.column_selector)
        filter_layout.addWidget(QLabel("Valor:"))
        filter_layout.addWidget(self.filter_input)
        filter_layout.addWidget(self.filter_button)
        filter_layout.addWidget(self.clear_filter_button)

        left_layout = QVBoxLayout()
        left_layout.addLayout(filter_layout)
        left_layout.addWidget(self.table_widget)

        right_layout = QVBoxLayout()
        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.plot_canvas)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=2)
        main_layout.addLayout(right_layout, stretch=3)

        self.table_widget.horizontalHeader().sectionClicked.connect(
            lambda index: self.show_sort_menu(index, self.plot_canvas, self.table_widget)
        )

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def update_table(self, data, table_widget):
        """
        Populate the QTableWidget with data from a pandas DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            The dataset to populate the table with.
        table_widget : QTableWidget
            The table widget to update.
        """
        num_rows, num_columns = data.shape
        table_widget.setRowCount(num_rows)
        table_widget.setColumnCount(num_columns)
        table_widget.setHorizontalHeaderLabels(data.columns)

        for row in range(num_rows):
            for col in range(num_columns):
                item = QTableWidgetItem(str(data.iat[row, col]))
                table_widget.setItem(row, col, item)

    def apply_filter(self, column_selector, filter_input, plot_canvas, table_widget):
        """
        Apply a filter to the dataset based on the selected column and filter text.

        Updates the table and plot accordingly.

        Parameters
        ----------
        column_selector : QComboBox
            ComboBox widget for selecting the column to filter by.
        filter_input : QLineEdit
            Text input for filter value.
        plot_canvas : PlotCanvas
            The plotting canvas to update.
        table_widget : QTableWidget
            The table to update with filtered data.
        """
        column = column_selector.currentText()
        value = filter_input.text()
        if value:
            filtered_data = self.dataset[self.dataset[column].astype(str).str.contains(value, na=False, case=False)]
            self.update_table(filtered_data, table_widget)
            plot_canvas.plot(filtered_data)
        else:
            self.update_table(self.dataset, table_widget)  # Restaurar el dataset original si no hay valor de filtro
            plot_canvas.plot(self.dataset)

    def clear_filter(self, filter_input, table_widget, plot_canvas, filter_button, column_selector):
        """
        Clear any applied filter and reset the table and plot to show the entire dataset.

        Parameters
        ----------
        filter_input : QLineEdit
            Text input to clear.
        table_widget : QTableWidget
            The table to reset.
        plot_canvas : PlotCanvas
            The plot canvas to reset.
        filter_button : QPushButton
            Filter button (not used directly here).
        column_selector : QComboBox
            Column selector (not used directly here).
        """
        filter_input.clear()
        self.update_table(self.dataset, table_widget)
        plot_canvas.plot(self.dataset)

    def show_sort_menu(self, column_index, plot_canvas, table_widget):
        """
        Show a context menu to sort the table by a selected column either ascending or descending,
        or to reset to original order.

        Parameters
        ----------
        column_index : int
            The index of the column header clicked.
        plot_canvas : PlotCanvas
            The canvas to update after sorting.
        table_widget : QTableWidget
            The table widget to update.
        """
        menu = QMenu()

        sort_asc_action = QAction("Orden Ascendente", menu)
        sort_desc_action = QAction("Orden Descendente", menu)
        reset_action = QAction("Mostrar Todo", menu)

        def sort_ascending():
            """
            Sort the dataset in ascending order by the selected column,
            update the table widget and re-plot the sorted data.

            This method sorts the main dataset based on the column at
            `column_index` in ascending order, updates the table to
            reflect this sorted data, and updates the plot accordingly.
            """
            sorted_data = self.dataset.sort_values(by=self.dataset.columns[column_index], ascending=True)
            self.update_table(sorted_data, table_widget)
            plot_canvas.plot(sorted_data)

        def sort_descending():
            """
            Sort the dataset in descending order by the selected column,
            update the table widget and re-plot the sorted data.

            This method sorts the main dataset based on the column at
            `column_index` in descending order, updates the table to
            reflect this sorted data, and updates the plot accordingly.
            """
            sorted_data = self.dataset.sort_values(by=self.dataset.columns[column_index], ascending=False)
            self.update_table(sorted_data, table_widget)
            plot_canvas.plot(sorted_data)

        def reset_table():
            """
            Reset the table widget and plot to show the original dataset
            without any sorting applied.

            This method reloads the original dataset into the table widget
            and re-plots the data to restore the initial view.
            """
            self.update_table(self.dataset, table_widget)
            plot_canvas.plot(self.dataset)

        sort_asc_action.triggered.connect(sort_ascending)
        sort_desc_action.triggered.connect(sort_descending)
        reset_action.triggered.connect(reset_table)

        menu.addAction(sort_asc_action)
        menu.addAction(sort_desc_action)
        menu.addAction(reset_action)

        header_pos = table_widget.mapToGlobal(table_widget.horizontalHeader().pos())
        menu.exec_(header_pos + QPoint(column_index * 100, table_widget.horizontalHeader().height()))

class PlotCanvas(FigureCanvas):
    """
    Matplotlib canvas integrated into PyQt5 for plotting dataset values.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure object for plotting.
    axes : matplotlib.axes.Axes
        The axes of the plot.

    Methods
    -------
    plot(df)
        Plot the first numeric column from the DataFrame, highlighting anomalies if present.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Initialize the plotting canvas.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget.
        width : float, optional
            Width of the figure in inches.
        height : float, optional
            Height of the figure in inches.
        dpi : int, optional
            Resolution of the figure.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot(self, df):
        """
        Plot the dataset’s first numeric column against timestamp or index.

        Anomalies are highlighted if the column 'is_anomaly' exists.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataset to plot.
        """
        self.axes.clear()
        df = df.copy()

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']) 
            x = df['timestamp']
            x_label = 'timestamp'
        else:
            x = df.index if isinstance(df.index, pd.DatetimeIndex) else range(len(df))
            x_label = 'Índice'

        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) == 0:
            self.axes.set_title("No hay columnas numéricas para graficar.")
            self.draw()
            return

        y_col = numeric_cols[0]
        y = df[y_col]

        self.axes.plot(x, y, marker='o', color='blue', label=y_col)
        if 'is_anomaly' in df.columns:
            anomalies = df[df['is_anomaly'] == 1]
            if not anomalies.empty:
                xa = anomalies['timestamp'] if 'timestamp' in anomalies.columns else anomalies.index
                ya = anomalies[y_col]
                self.axes.scatter(xa, ya, color='red', label='Anomalía', zorder=5)

        self.axes.set_title(f"Gráfico de '{y_col}' con anomalías")
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_col)
        self.axes.legend()
        self.axes.grid(True)

        if isinstance(x.iloc[0], pd.Timestamp):
            self.axes.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            self.fig.autofmt_xdate(rotation=45)

        self.draw()