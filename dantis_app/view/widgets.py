from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtWidgets import QListWidget
from PyQt5.QtGui import QDrag
from PyQt5.QtWidgets import QTableWidget,  QTableWidgetItem

class DropListWidget(QListWidget):
    """
    QListWidget subclass enabling drag and drop of its items.

    Attributes
    ----------
    _drag_item : QListWidgetItem or None
        Temporarily holds the item being dragged.

    Methods
    -------
    startDrag(supportedActions)
        Initiates a drag operation with the currently selected item.
    dragEnterEvent(event)
        Handles drag entering the widget area, accepting text mime data.
    dragMoveEvent(event)
        Accepts drag movement within the widget.
    dropEvent(event)
        Handles drop event to add new items from external drag sources.
    """
    def __init__(self, parent=None):
        """
        Initializes the DropListWidget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, by default None.
        """
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QListWidget.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)

    def startDrag(self, supportedActions):
        """
        Starts a drag operation for the currently selected item.

        Parameters
        ----------
        supportedActions : Qt.DropActions
            The supported drag actions.
        """
        self._drag_item = self.currentItem()
        if self._drag_item:
            drag = QDrag(self)
            mimeData = QMimeData()
            mimeData.setText(self._drag_item.text())
            drag.setMimeData(mimeData)

            result = drag.exec(Qt.MoveAction)
            if result == Qt.MoveAction and self._drag_item: 
                self.takeItem(self.row(self._drag_item))
                self._drag_item = None

    def dragEnterEvent(self, event):
        """
        Accepts drag entering event if the mime data contains text.

        Parameters
        ----------
        event : QDragEnterEvent
            The drag enter event.
        """
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        """
        Accepts drag move events to enable moving within the widget.

        Parameters
        ----------
        event : QDragMoveEvent
            The drag move event.
        """
        event.acceptProposedAction()

    def dropEvent(self, event):
        """
        Handles drop events by adding the dragged text as a new item,
        if it does not already exist in the list.

        Parameters
        ----------
        event : QDropEvent
            The drop event.
        """
        if event.source() == self:
            event.ignore()
            return

        if event.mimeData().hasText():
            text = event.mimeData().text()
            if not any(self.item(i).text() == text for i in range(self.count())):
                self.addItem(text)
            event.acceptProposedAction()

class DropTableWidget(QTableWidget):
    """
    QTableWidget subclass that accepts drag and drop from a linked QListWidget.

    Attributes
    ----------
    list_widget : QListWidget
        External list widget from which items can be dragged into the table.

    Methods
    -------
    startDrag(supportedActions)
        Starts a drag operation for the currently selected table item.
    dragEnterEvent(event)
        Accepts drag entering event if mime data has text.
    dragMoveEvent(event)
        Accepts drag move events within the table.
    dropEvent(event)
        Handles dropping text data into the table at the drop position.
    """
    def __init__(self, list_widget: QListWidget, parent=None):
        """
        Initializes the DropTableWidget and links it to a QListWidget.

        Parameters
        ----------
        list_widget : QListWidget
            The external list widget to interact with drag-and-drop.
        parent : QWidget, optional
            Parent widget, by default None.
        """
        super().__init__(parent)
        self.list_widget = list_widget
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QTableWidget.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)

    def startDrag(self, supportedActions):
        """
        Starts drag operation for the currently selected table item.

        Parameters
        ----------
        supportedActions : Qt.DropActions
            The supported drag actions.
        """
        item = self.currentItem()
        if item:
            mimeData = QMimeData()
            mimeData.setText(item.text())

            drag = QDrag(self)
            drag.setMimeData(mimeData)

            if drag.exec(Qt.MoveAction) == Qt.MoveAction:
                self.setItem(self.currentRow(), self.currentColumn(), None)

    def dragEnterEvent(self, event):
        """
        Accepts drag entering event if mime data contains text.

        Parameters
        ----------
        event : QDragEnterEvent
            The drag enter event.
        """
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        """
        Accepts drag move events within the table widget.

        Parameters
        ----------
        event : QDragMoveEvent
            The drag move event.
        """
        event.accept()

    def dropEvent(self, event):
        """
        Handles the drop event by inserting the dragged text into the cell
        at the drop position. If the cell already contains an item, it is
        returned to the linked list widget.

        Parameters
        ----------
        event : QDropEvent
            The drop event.
        """
        if event.mimeData().hasText():
            text = event.mimeData().text()
            pos = event.pos()
            row = self.rowAt(pos.y())
            col = self.columnAt(pos.x())

            if row == -1 or col == -1:
                return

            existing_item = self.item(row, col)
            if existing_item:
                self.list_widget.addItem(existing_item.text())

            self.setItem(row, col, QTableWidgetItem(text))
            event.acceptProposedAction()