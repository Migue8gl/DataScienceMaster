import warnings

warnings.filterwarnings("ignore")

import polars as pl

from lstm_utils import (
    evaluate,
    predict,
    train_lstm,
)
from utils import plot_confusion_matrix


def main():
    df = pl.read_csv("data/train.csv")
    test_df = pl.read_csv("data/test.csv")

    for category in df.select(pl.exclude("text")).columns:
        print(f"\nTraining model for category: {category}")

        X_train = df["text"].to_numpy()
        y_train = df[category].to_numpy()
        X_test = test_df["text"].to_numpy()
        y_test = test_df[category].to_numpy()

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        # Train the model
        model, _, label_decoder, vocab, tokenizer = train_lstm(
            X_train,
            y_train,
            hidden_size=128,
            num_layers=2,
            num_epochs=60,
            batch_size=32,
        )

        # Make predictions
        predictions = predict(model, X_test, tokenizer, vocab, label_decoder)

        # Print some prediction examples
        print("\nSample predictions:")
        for i in range(min(5, len(predictions))):
            print(f"True: {y_test[i]}, Predicted: {predictions[i]}")

        # Calculate accuracy
        accuracy = sum(
            str(predictions[i]) == str(y_test[i]) for i in range(len(y_test))
        ) / len(y_test)
        print(f"\nTest Accuracy for {category}: {accuracy:.4f}")

        # Plot confusion matrix
        try:
            plot_confusion_matrix(
                y_test,
                predictions,
                f"{category} confusion matrix",
                category,
                "lstm_per_level",
            )
            print(f"Confusion matrix plotted for {category}")
        except Exception as e:
            print(f"Could not plot confusion matrix: {str(e)}")

        # Evaluate the model
        try:
            save_kwargs = {"name": "lstm_per_level", "category": category}
            metrics_result = evaluate(
                model, X_test, y_test, tokenizer, vocab, label_decoder, **save_kwargs
            )
            print(f"Evaluation metrics for {category}:")
            print(metrics_result)
        except Exception as e:
            print(f"Note: Could not run evaluate function: {str(e)}")


if __name__ == "__main__":
    main()
