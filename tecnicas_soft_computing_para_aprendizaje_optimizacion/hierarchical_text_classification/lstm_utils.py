from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchtext
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm, trange

from utils import store_results


def create_vocab(dataset: np.ndarray, tokenizer: Any) -> torchtext.vocab:
    def yield_tokens(dataset):
        for text in dataset:
            if isinstance(text, list):
                for item in text:
                    yield tokenizer(item)
            else:
                yield tokenizer(text)

    vocab = build_vocab_from_iterator(
        yield_tokens(dataset),
        min_freq=2,
        specials=["<pad>", "<sos>", "<eos>", "<unk>"],
        special_first=True,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def predict(model, X, tokenizer, vocab, label_decoder):
    device = next(model.parameters()).device

    # Create a dataset for prediction (no labels needed)
    prediction_dataset = TextDataset(X, None, tokenizer, vocab)

    dataloader = DataLoader(
        prediction_dataset,
        batch_size=32,
        collate_fn=collate_batch_predict,
        shuffle=False,
    )

    predictions = []
    model.eval()

    with torch.no_grad():
        for text_batch in tqdm(dataloader, desc="Predicting"):
            text_batch = text_batch.to(device)
            outputs, _, _ = model(text_batch)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # Convert numeric predictions back to original labels
    return np.array([label_decoder[pred] for pred in predictions])


def evaluate(
    model, X_test, y_test, tokenizer=None, vocab=None, label_decoder=None, **save_kwargs
):
    """Evaluate the model and return metrics"""
    # Convert model predictions to numpy array
    if not isinstance(model, LSTMClassifier):
        y_pred = model.predict(X_test, save_kwargs["category"])
    else:
        y_pred = predict(model, X_test, tokenizer, vocab, label_decoder)

    # Calculate evaluation metrics
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = metrics.recall_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = metrics.precision_score(y_test, y_pred, average="weighted", zero_division=0)

    # Prepare results dictionary
    results = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # Print the evaluation metrics
    print(f"Accuracy: {100 * acc:.1f}%")
    print(f"Precision: {100 * prec:.1f}%")
    print(f"Recall: {100 * rec:.1f}%")
    print(f"F1-score: {100 * f1:.1f}%")

    store_results(
        **save_kwargs,
        metrics={"accuracy": acc, "f1": f1, "recall": rec, "precision": prec},
    )

    return results


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None, cell=None):
        # Initialize hidden states if not provided
        if hidden is None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
                x.device
            )
        if cell is None:
            cell = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
                x.device
            )

        # Apply embedding
        x = self.embedding(x)

        # Forward pass through LSTM
        out, (hidden, cell) = self.lstm(x, (hidden, cell))

        # Apply final fully connected layer to the last output
        out = self.fc(out[:, -1, :])
        return out, hidden, cell


class TextDataset(Dataset):
    def __init__(
        self, texts, labels, tokenizer, vocab, label_encoder=None, max_len=512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.max_len = max_len
        self.is_prediction = label_encoder is None or labels is None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize text
        tokens = self.tokenizer(text)

        # Convert tokens to indices
        token_ids = []
        for token in tokens:
            # Handle OOV tokens by using a default index or skipping
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Use the default unknown token index
                token_ids.append(self.vocab["<unk>"])

        # Truncate if necessary
        if len(token_ids) > self.max_len:
            token_ids = token_ids[: self.max_len]

        # If empty after filtering, add at least one token
        if len(token_ids) == 0:
            token_ids = [0]  # Use padding index

        # For prediction mode, we don't need labels
        if self.is_prediction:
            return {
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
            }
        else:
            label = str(
                self.labels[idx]
            )  # Convert to string to ensure consistent handling
            # Encode the label
            label_id = self.label_encoder[label]

            return {
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "labels": torch.tensor(label_id, dtype=torch.long),
            }


def create_label_encoder(labels):
    """Create a mapping from label strings to integer indices."""
    unique_labels = sorted(set(str(label) for label in labels))
    return {label: idx for idx, label in enumerate(unique_labels)}


def collate_batch_train(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    # Pad sequences to the same length
    input_ids = pad_sequence(input_ids, batch_first=True)

    return input_ids, labels


def collate_batch_predict(batch):
    input_ids = [item["input_ids"] for item in batch]

    # Pad sequences to the same length
    input_ids = pad_sequence(input_ids, batch_first=True)

    return input_ids


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_size: int = 64,
    num_layers: int = 2,
    lr: float = 0.001,
    num_epochs: int = 100,
    batch_size: int = 32,
) -> tuple:
    pbar = trange(0, num_epochs, leave=False, desc="Epoch")

    training_loss_logger = []
    eval_loss_logger = []
    training_acc_logger = []
    eval_acc_logger = []

    tokenizer = get_tokenizer("basic_english")
    vocab = create_vocab(X_train, tokenizer)
    vocab_size = len(vocab)

    # Create label encoder
    label_encoder = create_label_encoder(y_train)
    output_size = len(label_encoder)

    print(f"Number of unique labels: {output_size}")
    print(f"Vocabulary size: {vocab_size}")

    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = TextDataset(X_train, y_train, tokenizer, vocab, label_encoder)
    eval_dataset = TextDataset(X_eval, y_eval, tokenizer, vocab, label_encoder)

    data_loader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch_train,
        num_workers=4,
        drop_last=True,
    )
    data_loader_eval = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch_train,
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(vocab_size, hidden_size, num_layers, output_size).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in pbar:
        model.train()
        train_acc = 0
        train_samples = 0

        # Training loop
        for text_batch, labels_batch in tqdm(
            data_loader_train, desc="Training", leave=False
        ):
            text_batch = text_batch.to(device)
            labels_batch = labels_batch.to(device)

            batch_size = labels_batch.size(0)

            # Forward pass
            predictions, _, _ = model(text_batch)

            # Calculate loss
            loss = loss_fn(predictions, labels_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(predictions, 1)
            train_acc += (predicted == labels_batch).sum().item()
            train_samples += batch_size

            # Log training loss
            training_loss_logger.append(loss.item())

        # Calculate training accuracy for the epoch
        epoch_train_acc = train_acc / train_samples if train_samples > 0 else 0
        training_acc_logger.append(epoch_train_acc)

        # Evaluation loop
        model.eval()
        eval_acc = 0
        eval_samples = 0

        with torch.no_grad():
            for text_batch, labels_batch in tqdm(
                data_loader_eval, desc="Evaluating", leave=False
            ):
                text_batch = text_batch.to(device)
                labels_batch = labels_batch.to(device)

                batch_size = labels_batch.size(0)

                # Forward pass
                predictions, _, _ = model(text_batch)

                # Calculate loss
                loss = loss_fn(predictions, labels_batch)
                eval_loss_logger.append(loss.item())

                # Calculate accuracy
                _, predicted = torch.max(predictions, 1)
                eval_acc += (predicted == labels_batch).sum().item()
                eval_samples += batch_size

            # Calculate evaluation accuracy for the epoch
            epoch_eval_acc = eval_acc / eval_samples if eval_samples > 0 else 0
            eval_acc_logger.append(epoch_eval_acc)

        # Update progress bar with current accuracy
        pbar.set_postfix_str(
            f"Train Acc: {epoch_train_acc:.4f}, Eval Acc: {epoch_eval_acc:.4f}"
        )

    # Create inverse mapping for label predictions
    label_decoder = {idx: label for label, idx in label_encoder.items()}

    return model, label_encoder, label_decoder, vocab, tokenizer
