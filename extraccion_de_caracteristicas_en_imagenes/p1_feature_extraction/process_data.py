import argparse
import os
import textwrap
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import polars as pl

from LBPDescriptor import LBPDescriptor
from utils import create_descriptor


def process_image(
    file_path: str,
    image_size: Tuple[int, int],
    descriptor: Optional[Union[cv2.HOGDescriptor, LBPDescriptor]] = None,
) -> Optional[np.ndarray]:
    """
    Load and preprocess an image, returning its features or raw data.

    Args:
        file_path (str): Path to the image file
        image_size (Tuple[int, int]): Target image size
        descriptor (Optional): Feature descriptor to use
    Returns:
        np.ndarray: Processed image data
    """
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # One channel
    if img is None:
        print("Failed to load image. Ensure the file is supported and not corrupted.")
        return None
    img = img.astype(np.uint8)
    if img is None or img.size == 0:
        raise ValueError(f"Invalid image at {file_path}")
    img = cv2.resize(img, (image_size, image_size))

    result = img
    if descriptor is not None:
        result = descriptor.compute(img).flatten()
    return result


def save_dataset(
    path: str,
    labels: List[str],
    image_size: int,
    descriptor_params: Optional[Dict] = None,
    parquet_file: Optional[str] = None,
) -> bool:
    """
    Reads images from a dataset path, extracts descriptor features or raw data and save it.

    Args:
        path (str): Path to the dataset directory.
        labels (List[str]): List of label subdirectories.
        image_size (int): Target image size (MxM) for processing.
        descriptor_params (Optional[Dict]): Parameters for descriptor.
        parquet_file (Optional[str]): Path to save or load the dataset in Parquet format.
    Returns:
        bool: If the saving operation was successful
    """
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset path does not exist.")

    data, data_labels = [], []
    descriptor = None

    # Create descriptor if not raw data and descriptor params provided
    if descriptor_params:
        descriptor_type = descriptor_params.get("type", "hog")
        descriptor_config = descriptor_params.get("config", {})
        descriptor = create_descriptor(descriptor_config, descriptor_type, image_size)

    for label_idx, label_name in enumerate(labels):
        label_dir = os.path.join(path, label_name)
        if not os.path.exists(label_dir):
            raise FileNotFoundError(
                f"Data not found for label '{label_name}' at {label_dir}."
            )
        for filename in os.listdir(label_dir):
            file_path = os.path.join(label_dir, filename)
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                features = process_image(file_path, image_size, descriptor)
                if features is not None:
                    data.append(features)
                    data_labels.append(label_idx)

    # Convert to NumPy arrays
    data = np.array(data)
    data_labels = np.array(data_labels)

    # Shuffle
    p = np.random.permutation(data.shape[0])
    data, data_labels = data[p], data_labels[p]

    # Save as Parquet file using Polars if not raw data
    name = parquet_file if parquet_file is not None else "dataset"
    print(f"Saving processed dataset to {parquet_file} using Polars.")
    try:
        df = pl.DataFrame(
            {
                "features": data.tolist(),
                "label": data_labels.tolist(),
                "shape": [tuple(data.shape)] * len(data),
            }
        )
        df.write_parquet(name)
    except Exception as e:
        print(f"Error while writing to parquet: {e}")
        return False

    return True


def read_dataset(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads data from a parquet file and returns it in
    (data, labels) format.

    Args:
        data_path (str): Path where dataset file is stored.
    Returns (np.ndarray, np.ndarray): Data and labels.
    """
    # If Parquet file exists, load from file
    if os.path.exists(data_path):
        print(f"Loading dataset from {data_path} using Polars.")
        df = pl.read_parquet(data_path)
        data = np.array(df["features"].to_list())
        original_shape = df["shape"][0]  # Retrieve the original shape
        data = data.reshape(original_shape)
        data_labels = df["label"].to_numpy()
        return data, data_labels
    else:
        raise ValueError("Dataset path does not exists")


def main(args: argparse.Namespace):
    """Main functionality"""
    # Prepare parameters if a descriptor is used
    mode = args.mode.lower()
    descriptor_params = None
    if mode == "lbp":
        descriptor_params = {}
        descriptor_params["type"] = mode
        descriptor_params["config"] = {"radius": 2, "n_neighbors": 8}
    elif mode == "hog":
        descriptor_params = {}
        descriptor_params["type"] = mode
        descriptor_params["config"] = {
            "winSize": (args.image_size, args.image_size),
            "blockSize": (args.image_size // 2, args.image_size // 2),
            "blockStride": (args.image_size // 4, args.image_size // 4),
            "cellSize": (args.image_size // 4, args.image_size // 4),
            "nbins": 12,
            "derivAperture": 1,
            "winSigma": 5.0,
            "histogramNormType": 0,
            "L2HysThreshold": 0.3,
            "gammaCorrection": 0,
            "nlevels": 64,
            "signedGradients": True,
        }
    # Parquet file naming strategy
    parquet_file = (
        f"dataset_{mode}_{"_".join(args.labels).lower()}_{args.image_size}.parquet"
    )

    # Save dataset in parquet file
    save_dataset(
        args.data_path, args.labels, args.image_size, descriptor_params, parquet_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """This script aims to preprocess and save a dataset into a efficient file format."""
        )
    )
    parser.add_argument(
        "-d", "--data-path", type=str, default="./data", help="Path to the dataset."
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        nargs="+",
        default=["Cat", "Dog"],
        help="Labels for the dataset.",
    )
    parser.add_argument(
        "-s",
        "--image-size",
        type=int,
        default=28,
        help="Image size (MxM) to resize.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["lbp", "hog", "raw"],
        default="raw",
        help="Mode of processing: 'lbp', 'hog', or 'raw'. Default is 'raw'.",
    )
    args = parser.parse_args()
    main(args)
