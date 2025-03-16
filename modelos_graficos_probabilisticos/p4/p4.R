library(bnlearn)
library(Rgraphviz)
library(gRain)
library(gridExtra)
library(tidyverse)
library(corrplot)
library(mRMRe)

# Load ARFF data, skipping metadata lines starting with @
data <- read.csv("datos/ledLXMn10.arff",
                 header = FALSE,
                 comment.char = "@")

# Initial data inspection
str(data)
head(data)
cat("Number of instances:", nrow(data), "\n")
cat("Number of variables:", ncol(data), "\n")

# Exploratory Data Analysis (EDA)
summary(data)
cat("\nMissing values per column:\n")
print(colSums(is.na(data)))

# Calculate correlation matrix for numerical features
cor_matrix <- cor(data)
corrplot(cor_matrix, method = "number", type = "upper")

# Check feature variance
feature_variance <- sapply(data, function(x)
  var(as.numeric(as.character(x))))
cat("\nFeature variances:\n")
print(feature_variance)

# Convert all features to factors for bnlearn compatibility
data <- data %>% mutate(across(everything(), ~ factor(., ordered = TRUE)))
str(data)

# Feature selection
file.data <- mRMR.data(data = data.frame(data))
results <- mRMR.classic(
  "mRMRe.Filter",
  data = file.data,
  target_indices = 25,
  feature_count = 10
)
selected_features <- solutions(results)
filtered_data <- data[, c(selected_features[[1]], 25)]
str(filtered_data)

# Create train/test split (80/20)
set.seed(123)
train_indices <- sample(1:nrow(filtered_data), 0.8 * nrow(filtered_data))
train <- filtered_data[train_indices, ]
test <- filtered_data[-train_indices, ]

features <- setdiff(names(filtered_data), "V25")
blacklist <- tiers2blacklist(list(features, "V25"))

# Try different structure learning approaches
models <- list(
  hc = hc(train, blacklist = blacklist, score = "k2"),
  tabu = tabu(train, blacklist = blacklist),
  iamb = cextend(iamb(train, blacklist = blacklist))
)

# Fit models with class variable
fitted_models <- lapply(models, function(m) {
  bn.fit(m, data = train)
})

# Evaluation function
evaluate_model <- function(fitted_model, test_data) {
  pred <- predict(fitted_model, node = "V25", data = test_data)
  confusion <- table(pred, test_data$V25)
  accuracy <- sum(diag(confusion)) / sum(confusion)
  
  # Calculate additional metrics
  precision <- diag(confusion) / rowSums(confusion)
  recall <- diag(confusion) / colSums(confusion)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  specificity <- sapply(1:nrow(confusion), function(i) {
    tn <- sum(confusion[-i, -i])
    fp <- sum(confusion[i, -i])
    tn / (tn + fp)
  })
  
  list(
    accuracy = accuracy,
    confusion = confusion,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    specificity = specificity
  )
}

# Evaluate models with detailed results
results <- lapply(fitted_models, evaluate_model, test)

# Print comprehensive results
cat("\nModel Performance:\n")
for (model_name in names(results)) {
  cat(paste0("\n", toupper(model_name), ":\n"))
  cat("Accuracy:", round(results[[model_name]]$accuracy, 3), "\n")
  cat("Confusion Matrix:\n")
  print(results[[model_name]]$confusion)
  cat("Precision:\n")
  print(results[[model_name]]$precision)
  cat("Recall:\n")
  print(results[[model_name]]$recall)
  cat("F1-Score:\n")
  print(results[[model_name]]$f1_score)
  cat("Specificity:\n")
  print(results[[model_name]]$specificity)
}

for (model_name in names(models)) {
  # Plot the structure
  graphviz.plot(models[[model_name]], )
}
