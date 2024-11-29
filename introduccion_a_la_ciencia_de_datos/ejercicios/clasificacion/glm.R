library(philentropy)
library(tidyr)
library(dplyr)
library(ggplot2)
library(caret)

# Read csv
data <- read.csv("cancer_dataset.csv")

# Look for examples in data
head(data)

# Some statistics for the data
summary(data)

# Get random indexes to split data into test and train. Train will be 80% of the
# total amount of data, the other 20% is going to be in test. Also, we need to
# extract the labels.
set.seed(42) # For reproducibility
indexes <- sample(seq_len(nrow(data)))
train_size <- as.integer(nrow(data) * 0.8)
indexes_train <- indexes[1:train_size]
indexes_test <- indexes[(train_size + 1):nrow(data)]
train <- data[indexes_train, 3:ncol(data)]
train_labels <- as.numeric(factor(data[indexes_train, 2], levels = unique(data[, 2]))) - 1
test <- data[indexes_test, 3:ncol(data)]
test_labels <- as.numeric(factor(data[indexes_test, 2], levels = unique(data[, 2]))) - 1

# Define a k-fold cross-validation function
k_fold_cross_validation <- function(x, x_labels, k = 10) {
  # Shuffle train data
  idx <- sample(nrow(x))
  x <- x[idx, ]
  x_labels <- x_labels[idx]
  n <- round(nrow(x) / k)
  best_model <- NA
  best_accuracy <- -Inf
  
  # Initialize variables to store performance metrics
  accuracy_list <- c()
  
  for (i in 0:(k - 1)) {
    start <- 1 + i * n
    end <- min(n + i * n, nrow(x)) # Ensure 'end' does not exceed data size
    
    val <- x[start:end, ]
    val_labels <- x_labels[start:end]
    train_cv <- x[-(start:end), ]
    train_cv_labels <- x_labels[-(start:end)]
    
    # Train the model on k-1 folds
    model <- glm(
      train_cv_labels ~ .,
      data = data.frame(train_cv, train_cv_labels),
      family = binomial,
    )
    
    # Predict on the validation fold
    predicted_probs <- predict(model, val)
    predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)
    
    # Calculate accuracy for this fold
    accuracy <- mean(predicted_classes == val_labels)
    accuracy_list <- c(accuracy_list, accuracy)
    
    # Update best model
    if (best_accuracy < accuracy) {
      best_accuracy <- accuracy
      best_model <- model
    }
    
    print(paste("Fold", i + 1, "Accuracy:", round(accuracy, 4)))
  }
  
  # Output the average accuracy across all folds
  avg_accuracy <- mean(accuracy_list)
  print(paste("Average Accuracy:", round(avg_accuracy, 4)))
  
  # Return best model
  best_model
}

# Run k-fold cross-validation
glm_model <- k_fold_cross_validation(train, train_labels)

# Check model performance in test set
predicted_probs <- predict(glm_model, test)
predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)

print(paste("Test Accuracy: ", round(mean(
  predicted_classes == test_labels
), 4)))
