import numpy as np
import logging


def clean_json(obj):
    """
    Recursively converts NumPy types and arrays to native Python types
    so that the object can be serialized into JSON.

    Parameters
    ----------
    obj : any
        The object to be cleaned. Can be a dict, list, NumPy type, or other.

    Returns
    -------
    any
        A cleaned version of the object with only JSON-serializable native types.
    """
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def format_info_results(results: dict) -> str:
    """
    Formats a dictionary of training results into a clean list of dataset reports.

    Parameters
    ----------
    results : dict
        Dictionary containing model training results for each dataset.

    Returns
    -------
    list
        A list of dictionaries with formatted training information for reporting/export.
    """
    details = []

    for idx, result in results.items():
        dataset_info = result.get('dataset')

        dataset_entry = {
            "dataset_id": result.get('dataset_id') + 1
        }

        if isinstance(dataset_info, dict):
            if dataset_info.get('val'): 
                dataset_entry["nombre_dataset"] = f"{dataset_info.get('train').name}, {dataset_info.get('val').name} y {dataset_info.get('test').name}"
            else: 
                dataset_entry["nombre_dataset"] = f"{dataset_info.get('train').name} y {dataset_info.get('test').name}"
        else:
            dataset_entry["nombre_dataset"] = dataset_info.name

        dataset_entry["modelo"] = result.get("model", "Desconocido")
        dataset_entry["path_modelo"] = result.get("model_path", "[No especificado]")
        dataset_entry["hiperparametros"] = result.get("hyperparameters", {})

        if result.get("error"): 
            dataset_entry["error"] = "No se ha realizado entrenamiento. Validación insuficiente."
        else: 
            folds = []
            for fold in result.get("folds", []):
                fold_info = {
                    "fold": fold.get("fold", "?"),
                    "info": fold.get("info", None),
                    "scores": fold.get("scores", []),
                    "predictions": fold.get("predictions", []),
                    "metrics": fold.get("metrics", {})
                }
                folds.append(fold_info)
            dataset_entry["folds"] = folds

            test = result.get("test", {})
            dataset_entry["test"] = {
                "scores": test.get("scores", []),
                "predictions": test.get("predictions", []),
                "metrics": test.get("metrics", {})
            }

        details.append(dataset_entry)
    return details


def format_training_info(datasets, x_cols, y_cols,
                            models, metrics, val_opts, threshold) -> str:
    """
    Generates an HTML-formatted summary of the training configuration.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets, either complete or partitioned (train/test/val).
    x_cols : dict
        Dictionary of selected input columns per dataset.
    y_cols : dict
        Dictionary of selected output columns per dataset.
    models : dict
        Dictionary containing selected models and their parameters.
    metrics : list
        List of evaluation metric names.
    val_opts : dict
        Dictionary of validation options (e.g., cross-validation settings).
    threshold : float
        Threshold value used in training or evaluation.

    Returns
    -------
    str
        An HTML string representing the training configuration summary.
    """
    lines = []
    lines.append("<h2>DATOS PARA ENTRENAMIENTO:</h2>")
    
    full_datasets = {idx: ds for idx, ds in datasets.items() if not isinstance(ds, dict)}
    split_datasets = {idx: ds for idx, ds in datasets.items() if isinstance(ds, dict)}

    # Full datasets
    if full_datasets:
        lines.append("<b>Datasets completos</b>:<br>")
        for idx, info in full_datasets.items():
            lines.append(f"<u>Dataset {idx + 1}: {info.name}</u><br>")
            inputs = ", ".join([k for k, v in x_cols[idx].items() if v])
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>Entrada/s:</b> {inputs}<br>")
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>Salida/s:</b> {y_cols[idx]}<br><br>")
        lines.append("<br>")

    # Partitioned datasets
    if split_datasets:
        lines.append("<b>Datasets particionados</b>:<br>")
        for idx, parts in split_datasets.items():
            lines.append(f"<u>Dataset {idx + 1}:</u><br>")
            for split_name, info in parts.items():
                lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;<b>{split_name}:</b> {info.name}<br>")
            inputs = ", ".join(x_cols[idx].keys())
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Entrada/s:</b> {inputs}<br>")
            lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Salida/s:</b> {y_cols[idx]}<br><br>")
        lines.append("<br>")

    lines.append("<b>Modelos</b>:<br>")
    for m in models.values():
        lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;- {m['model']}<br>")
    lines.append("<br>")

    lines.append("<b>Métricas</b>:<br>")
    for met in metrics:
        lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;- {met}<br>")
    lines.append("<br>")

    lines.append("<b>Opciones validación:</b><br>")
    for k, v in val_opts.items():
        lines.append(f"&nbsp;&nbsp;&nbsp;&nbsp;- {k}: {v}<br>")
    lines.append("<br>")

    lines.append(f"<b>Threshold</b>: {threshold}<br>")
    lines.append("<hr>")

    return "\n".join(lines)


def format_results(id, results: dict) -> str:
    """
    Formats training and evaluation results as an HTML string for a given dataset.

    Parameters
    ----------
    id : int
        Identifier of the dataset (not used internally).
    results : dict
        Dictionary containing results for training, validation (folds), and test evaluation.

    Returns
    -------
    str
        An HTML-formatted string describing the training outcome and evaluation metrics.
    """
    text = [f"<b>Resultados para Dataset {results.get('dataset_id') + 1} con modelo {results.get('model')}</b><br><br>"]
    logging.debug("results: ", results)
    if results.get("error"): 
        text.append("Datos de validación introducidos insuficientes. No se ha realizado el entrenamiento.")
        return "\n".join(text)

    text.append("--- Validación Cruzada (Folds) ---<br>")
    folds = results.get("folds", [])
    if any(f.get("metrics") for f in folds): 
        for fold in folds:
            text.append(f"&nbsp;Fold {fold['fold']}:<br>")
            text.append(f"&nbsp;&nbsp;Métricas:<br>")
            metrics = fold.get("metrics")
            if metrics:
                for key, value in metrics.items():
                    text.append(f"&nbsp;&nbsp;&nbsp;- {key}: {value}<br>")
            else:
                text.append("&nbsp;&nbsp;&nbsp;- No se calcularon métricas para este fold<br>")
    else:
        text.append("&nbsp;&nbsp;- No se aplicó validación (folds vacíos) ---<br>")

    text.append("<br>")
    text.append("--- Evaluación con Test Final ---<br>")
    test_metrics = results.get("test", {}).get("metrics", {})
    text.append("&nbsp;Métricas:<br>")
    
    if test_metrics:
        for key, value in test_metrics.items():
            text.append(f"&nbsp;&nbsp;- {key}: {value}<br>")
    else:
        text.append("&nbsp;&nbsp;- No se calcularon métricas para el test<br>")

    return "\n".join(text)