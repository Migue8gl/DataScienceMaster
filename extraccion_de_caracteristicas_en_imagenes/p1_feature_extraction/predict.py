import argparse
import textwrap
from typing import Dict, Optional, Union

import cv2
from sklearn.base import BaseEstimator

from LBPDescriptor import LBPDescriptor
from utils import create_descriptor, load_model


def classify_image(
    model: BaseEstimator,
    image_path: str,
    descriptor: Union[LBPDescriptor, cv2.HOGDescriptor],
    preprocess_params: Optional[Dict] = None,
) -> Union[str, int]:
    """
    Classify a given image using a pre-trained model and descriptor.

    Args:
        model (BaseEstimator): Pre-trained classification model.
        image_path (str): Path to the image to be classified.
        descriptor (Union[LBPDescriptor, cv2.HOGDescriptor]): Feature descriptor used for extracting features.
        preprocess_params (Optional[Dict]): Optional parameters for preprocessing the image.

    Returns:
        Union[str, int]: Predicted class label or index.
    """
    try:
        # Load and preprocess image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        if preprocess_params:
            image_size = preprocess_params.get("image_size", 28)
            image = cv2.resize(image, (image_size, image_size))

        # Extract features using the descriptor
        if isinstance(descriptor, cv2.HOGDescriptor):
            features = descriptor.compute(image).flatten()
        elif isinstance(descriptor, LBPDescriptor):
            features = descriptor.compute(image).flatten()
        else:
            raise ValueError("Unsupported descriptor type")

        # Reshape features for model prediction
        features = features.reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        return prediction[0]

    except Exception as e:
        print(f"Error classifying image: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""Script for predictions""")
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        required=True,
        help="Path to the directory containing the pretrained model.",
    )

    parser.add_argument(
        "-i",
        "--image-path",
        type=str,
        required=True,
        help="Path to the image to be classified.",
    )
    parser.add_argument(
        "-s",
        "--image_size",
        type=int,
        default=28,
        help="Image size for HOG-based descriptors. Default is 128.",
    )
    parser.add_argument(
        "-desc",
        "--descriptor",
        type=str,
        choices=["lbp", "hog"],
        default=None,
        help="When doing hyperparameter search for a descriptor it should be specified.",
    )

    args = parser.parse_args()

    model, metadata = load_model(args.model_path)
    descriptor = create_descriptor(
        params={}, descriptor_type=args.descriptor, image_size=args.image_size
    )
    
    predicted_class = classify_image(
        model, args.image_path, descriptor, {"image_size":args.image_size}
    )
    print(f"Predicted Class: {predicted_class}")
