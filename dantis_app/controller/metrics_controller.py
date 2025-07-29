class MetricsController:
    """
    Controller class to manage metric selection via checkboxes and simulate metric values.

    This class stores a list of selected metrics represented by checkbox names 
    and provides methods to set and get these selections. It also includes a method 
    to simulate metric values for testing or demonstration purposes.

    Attributes
    ----------
    checkboxes : list of str
        List containing the names of the selected metric checkboxes.

    Methods
    -------
    setChecBoxesSelected(checboxes)
        Set the current list of selected checkboxes.

    getChecBoxes_selected()
        Get the current list of selected checkboxes.

    """
    def __init__(self):
        """
        Initialize a new MetricsController instance.

        Sets up an empty list to hold selected checkboxes.
        """
        self.checkboxes = []

    def setChecBoxesSelected(self, checboxes): 
        """
        Set the current list of selected checkboxes.

        Parameters
        ----------
        checboxes : list of str
            A list containing the names of the selected metric checkboxes.
        """
        self.checkboxes = checboxes

    def getChecBoxes_selected (self):
        """
        Get the current list of selected checkboxes.

        Returns
        -------
        list of str
            The list of currently selected metric checkbox names.
        """
        return self.checkboxes
