import logging
import os
from typing import Any, List, Tuple, Dict, Optional

import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils import plot_confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class TextLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.3
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output


class HierarchicalLSTMModel:
    def __init__(self, classifiers: List[Any], hierarchy_labels: List[str]):
        """
        Initialize hierarchical model with classifiers and their labels
        """
        self.classifiers = classifiers
        self.hierarchy_labels = hierarchy_labels

    def predict(self, X: List[str], level_label: Optional[str] = None) -> List[str]:
        """
        Make hierarchical predictions
        """
        # If no level is specified, default to the last (deepest) level
        if level_label is None:
            level_label = self.hierarchy_labels[-1]

        # Root level prediction
        root_classifier = self.classifiers[0]
        current_pred = self._predict_single_model(root_classifier, X)

        # If root level is target, return predictions
        if level_label.lower() == self.hierarchy_labels[0].lower():
            return current_pred

        # Iterate through subsequent levels
        for level, lvl_label in zip(self.classifiers[1:], self.hierarchy_labels[1:]):
            new_pred = []
            for i, pred in enumerate(current_pred):
                # Get child classifier for current parent prediction
                classifier_dict = level if isinstance(level, dict) else None

                if classifier_dict:
                    child_classifier = classifier_dict.get(pred.lower())
                    if child_classifier:
                        # Predict using child classifier
                        child_pred = self._predict_single_model(
                            child_classifier, [X[i]]
                        )
                        new_pred.append(child_pred[0])
                    else:
                        # Fallback to parent prediction if no child classifier
                        new_pred.append(pred)
                else:
                    new_pred.append(pred)

            current_pred = new_pred

            # Stop if we've reached the desired level
            if lvl_label.lower() == level_label.lower():
                break

        return current_pred

    def _predict_single_model(self, model, X: List[str]) -> List[str]:
        """
        Helper method to predict using a single LSTM model
        """
        # Preprocess text
        X_numerical, _ = self._preprocess_text(X, model["vocabulary"])
        X_tensor = torch.LongTensor(X_numerical)

        # Load model and set to evaluation mode
        lstm_model = TextLSTM(
            vocab_size=len(model["vocabulary"]) + 1,
            embedding_dim=100,
            hidden_size=128,
            num_classes=len(model["label_encoder"].classes_),
        )
        lstm_model.load_state_dict(torch.load(model["model_path"]))
        lstm_model.eval()

        # Predict
        with torch.no_grad():
            outputs = lstm_model(X_tensor)
            predictions = model["label_encoder"].inverse_transform(
                torch.argmax(outputs, dim=1).numpy()
            )

        return predictions.tolist()

    def _preprocess_text(
        self, texts: List[str], vocabulary: Dict[str, int], max_length: int = 100
    ):
        """
        Tokenize and convert texts to numerical sequences
        """
        # Simple tokenization
        tokens = [text.lower().split()[:max_length] for text in texts]

        # Convert to numerical sequences
        numerical_texts = [
            [vocabulary.get(word, 0) for word in text] + [0] * (max_length - len(text))
            for text in tokens
        ]

        return np.array(numerical_texts), vocabulary


def train_hierarchy(
    df: pl.DataFrame, hierarchy: List[Tuple[Optional[str], str]]
) -> HierarchicalLSTMModel:
    """
    Train hierarchical LSTM model
    hierarchy: List of tuples specifying (filter_column, target_column)
    """
    classifiers = []
    current_df = df
    model_save_dir = "lstm_model_checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)

    for level, (filter_col, target_col) in enumerate(hierarchy):
        if level == 0:  # Root level
            X_train = current_df["text"].to_list()
            y_train = current_df[target_col].to_list()

            # Preprocess text
            X_numerical, vocabulary = _preprocess_text(X_train)

            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_train)

            # Train LSTM model
            model, model_path = _train_lstm_model(
                X_numerical,
                y_encoded,
                vocab_size=len(vocabulary) + 1,
                num_classes=len(np.unique(y_encoded)),
                model_save_dir=model_save_dir,
                level=target_col,
            )

            # Store model information
            classifiers.append(
                {
                    "model": model,
                    "model_path": model_path,
                    "vocabulary": vocabulary,
                    "label_encoder": label_encoder,
                }
            )
        else:
            # Subsequent levels
            level_classifiers = {}
            parent_categories = current_df[filter_col].unique().to_list()

            for category in parent_categories:
                filtered_df = current_df.filter(pl.col(filter_col) == category)
                X_train = filtered_df["text"].to_list()
                y_train = filtered_df[target_col].to_list()

                # Only train if multiple classes exist
                if len(np.unique(y_train)) > 1:
                    # Preprocess text
                    X_numerical, vocabulary = _preprocess_text(X_train)

                    # Encode labels
                    label_encoder = LabelEncoder()
                    y_encoded = label_encoder.fit_transform(y_train)

                    # Train LSTM model
                    model, model_path = _train_lstm_model(
                        X_numerical,
                        y_encoded,
                        vocab_size=len(vocabulary) + 1,
                        num_classes=len(np.unique(y_encoded)),
                        model_save_dir=model_save_dir,
                        level=f"{filter_col}_{category}_{target_col}",
                    )

                    # Store model for this category
                    level_classifiers[category.lower()] = {
                        "model": model,
                        "model_path": model_path,
                        "vocabulary": vocabulary,
                        "label_encoder": label_encoder,
                    }

            classifiers.append(level_classifiers)

            # Filter dataframe to include only trained parent categories
            current_df = current_df.filter(pl.col(filter_col).is_in(parent_categories))

    # Extract labels from hierarchy
    hierarchy_labels = [cat[1] for cat in hierarchy]

    return HierarchicalLSTMModel(classifiers, hierarchy_labels)


def _preprocess_text(texts: List[str], max_length: int = 100):
    """
    Tokenize and create vocabulary
    """
    # Simple tokenization
    tokens = [text.lower().split()[:max_length] for text in texts]

    # Create vocabulary
    vocab = sorted(set(word for text in tokens for word in text))
    vocab_to_int = {word: i + 1 for i, word in enumerate(vocab)}

    # Convert to numerical sequences
    numerical_texts = [
        [vocab_to_int.get(word, 0) for word in text] + [0] * (max_length - len(text))
        for text in tokens
    ]

    return np.array(numerical_texts), vocab_to_int


def _train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    vocab_size: int,
    num_classes: int,
    model_save_dir: str,
    level: str,
    embedding_dim: int = 100,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_epochs: int = 60,
    batch_size: int = 32,
):
    """
    Train LSTM model and save
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.LongTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.LongTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = TextLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers,
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    best_val_accuracy = 0
    model_filename = f"{level}_best_model.pth"
    model_path = os.path.join(model_save_dir, model_filename)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()

        val_accuracy = val_correct / len(y_val)
        logger.info(
            f"Level {level}, Epoch {epoch + 1}, Val Accuracy: {val_accuracy:.4f}"
        )

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_path)

    return model, model_path


def main():
    # Load data
    df = pl.read_csv("data/train.csv")
    test_df = pl.read_csv("data/test.csv")

    # Define hierarchy
    categories = df.select(pl.exclude("text")).columns
    hierarchy = [(None, categories[0])] + [
        (categories[i], categories[i + 1]) for i in range(len(categories) - 1)
    ]

    # Train hierarchical model
    model = train_hierarchy(df, hierarchy)

    # Predict and evaluate for each category
    for category in categories:
        X_test = test_df["text"].to_list()
        y_test = test_df[category].to_list()

        # Predict using the hierarchical model
        predictions = model.predict(X_test, category)

        # Compute accuracy
        accuracy = np.mean([p == t for p, t in zip(predictions, y_test)])
        logger.info(f"Accuracy for {category}: {accuracy:.4f}")

        # Plot confusion matrix
        try:
            plot_confusion_matrix(
                y_test,
                predictions,
                f"{category} confusion matrix",
                category,
                "lstm_per_parent",
            )
            logger.info(f"Confusion matrix plotted for {category}")
        except Exception as e:
            logger.error(f"Could not plot confusion matrix for {category}: {str(e)}")


if __name__ == "__main__":
    main()
