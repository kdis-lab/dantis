from PyQt5.QtWidgets import QFileDialog
import json, zipfile, io

from controller.results_formatter import clean_json, format_info_results
from core.utils import show_error

def download_data_click(results):
    """
    Saves model training results to a JSON file.

    Opens a file dialog for the user to choose the save location.
    The results are formatted and cleaned before being written to disk.

    Parameters
    ----------
    results : dict
        Dictionary containing the model training results to be saved.

    Returns
    -------
    None
        The function returns nothing. If there are no results or the user cancels the operation,
        the function exits silently.
    """
    if not results:
        show_error("No training data has been provided for model training.")
        return

    archivo_json, _ = QFileDialog.getSaveFileName(
        None,
        "Save Report as JSON",
        "",
        "JSON Files (*.json)"
    )

    if not archivo_json:
        return

    report  = {
        "training_report": format_info_results(results)
    }
    report  = clean_json(report )

    with open(archivo_json, mode='w', encoding='utf-8') as f:
        json.dump(report , f, indent=4, ensure_ascii=False)
    
def download_data_test_click(statistical_results):
    """
    Saves statistical test results into a ZIP archive.

    The ZIP file contains a JSON with cleaned results and PNG images
    of any associated matplotlib figures.

    Parameters
    ----------
    statistical_results : dict
        Dictionary containing statistical test results. Each key is a metric name
        and the value is a dictionary with result data, possibly including
        matplotlib figures under the "_figures" key.

    Returns
    -------
    None
        The function returns nothing. If no results are available or the user cancels,
        it exits silently. Errors during the saving process are displayed using `show_error`.
    """
    if not statistical_results:
        show_error("Run the statistical tests before attempting to download the results.")
        return

    filename, _ = QFileDialog.getSaveFileName(
        None,
        "Save Results as ZIP",
        "",
        "Archivo ZIP (*.zip)"
    )

    if not filename:
        return

    if not filename.endswith(".zip"):
        filename += ".zip"

    try:
        with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            cleaned_results = {}

            for metric, result in statistical_results.items():
                result_copy = {
                    k: v for k, v in result.items() if not k.startswith("_")
                }

                cleaned_results[metric] = result_copy

                figures = result.get("_figures", [])
                for i, fig in enumerate(figures):
                    img_bytes = io.BytesIO()
                    fig.savefig(img_bytes, format='png', bbox_inches='tight')
                    img_bytes.seek(0)
                    fig_name = f"figures/{metric}.png"
                    zipf.writestr(fig_name, img_bytes.read())

            json_str = json.dumps(cleaned_results, indent=4, ensure_ascii=False)
            zipf.writestr("statistical_results.json", json_str)

    except Exception as e:
        show_error(f"Error al guardar el archivo ZIP: {e}")