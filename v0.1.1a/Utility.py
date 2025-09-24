from pathlib import Path
import cv2
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np


def convet2jpg(input_path: Path,
               output_path: Path) -> None:
    """
    Convert all images in a directory to JPEG (.jpg) format.

    This function reads all files from the `input_path` directory, converts
    images with an alpha channel (4 channels) to standard BGR (3 channels),
    and saves them in JPEG format to `output_path`. The output directory will
    be created if it does not exist.

    :param input_path: Path to the directory containing source images.
    :param output_path: Path to the directory where converted JPEG images will be saved.
    :return: None
    :notes:
        - Non-image files or unreadable files are skipped.
        - Images already in 3-channel BGR format are just saved as .jpg.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    for image_path in input_path.glob("*"):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None: continue

        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        output = output_path / (image_path.stem + ".jpg")
        cv2.imwrite(str(output), image)

def clustering_metrics(embeddings: np.ndarray,
                       labels: np.ndarray,
                       method_name: str) -> None:
    """
    Compute and display clustering evaluation metrics.

    This function calculates Silhouette Score and Davies-Bouldin Index to
    evaluate the quality of clustering for a given set of embeddings and
    cluster labels. It prints the results to the console.

    :param embeddings: 2D array of shape (n_samples, n_features) representing the feature embeddings of the dataset.
    :param labels: 1D array of cluster labels assigned to each embedding.
    :param method_name: Name or description of the clustering method (used for display purposes).
    :return: None
    :notes:
        - Silhouette Score ranges from -1 to 1; higher is better.
        - Davies-Bouldin Index is non-negative; lower is better.
        - If all samples are assigned to a single cluster, metrics cannot be computed.
    """
    if (len(labels) > 1):
        silhouette = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)

        print(f"Method: {method_name} | Silhouette: {silhouette} | Davies-Bouldin: {davies_bouldin}")
    else:
        print(f"Method: {method_name} | Error: Only one cluster detected!")