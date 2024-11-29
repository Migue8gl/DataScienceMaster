library(philentropy)
library(tidyr)
library(dplyr)
library(ggplot2)

my_knn <- function(train,
                   train_labels,
                   test,
                   k = 1,
                   metric = "euclidean") {
  # Convert train and test datasets to numeric matrices
  train <- as.matrix(sapply(train, as.numeric))
  test <- as.matrix(sapply(test, as.numeric))
  predictions <- rep(NA, nrow(test))
  
  # Calculate the distance matrix using the distance function
  distances <- distance(rbind(test, train), metric)
  
  # Separate the distances for test and train
  test_distances <- distances[1:nrow(test), (nrow(test) + 1):ncol(distances)]
  
  for (i in 1:nrow(test)) {
    neighbor_indices <- order(test_distances[i, ])[1:k]
    neighbor_labels <- train_labels[neighbor_indices]
    
    # Get the most common label among neighbors
    predictions[i] <- names(sort(table(neighbor_labels), decreasing = TRUE))[1]
  }
  
  predictions
}

# Read csv
data <- read.csv("cancer_dataset.csv")

# Look for examples in data
head(data)

# Some statistics for the data
summary(data)

# Search for NA values in dataset
data %>%
  summarize(across(everything(), ~ sum(is.na(.))))

# Get random indexes to split data into test and train. Train will be 80% of the
# total amount of data, the other 20% is going to be in test. Also, we need to
# extract the labels.
indexes <- sample(nrow(data))
train_size <- as.integer(nrow(data) * 0.8)
indexes_train <- indexes[1:train_size]
indexes_test <- indexes[(train_size + 1):nrow(data)]
train <- data[indexes_train, 3:length(data)]
train_labels <- as.factor(data[indexes_train, 2])
test <- data[indexes_test, 3:length(data)]
test_labels <- as.factor(data[indexes_test, 2])

# Compute knn experiments
distances = c("manhattan", "euclidean")
k_numbers <- seq(1, 50, 5)
accuracies <- data.frame(matrix(NA, ncol = 2, nrow = length(k_numbers)))
row.names(accuracies) <- k_numbers
colnames(accuracies) <- distances

for (i_distance in seq_along(distances)) {
  for (i in seq_along(k_numbers)) {
    predicted_labels <- my_knn(
      train = train,
      train_labels = train_labels,
      test = test,
      k = k_numbers[i],
      metric = distances[i_distance]
    )
    accuracies[i, i_distance] <- mean(predicted_labels == test_labels)
  }
}

accuracies_plot <- accuracies
accuracies_plot$k <- as.numeric(row.names(accuracies_plot))

# Reshaping data from wide to long format
data_long <- pivot_longer(
  accuracies_plot,
  cols = distances,
  names_to = "metric",
  values_to = "accuracy"
)

# Plotting
ggplot(data_long, aes(x = k, y = accuracy, color = metric)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = c(
    "manhattan" = "red",
    "euclidean" = "lightblue"
  )) +
  labs(
    title = "Comparison of Manhattan and Euclidean Distance Metrics over K",
    x = "k",
    y = "Accuracy",
    color = "Distance Metric"
  ) +
  theme_minimal()