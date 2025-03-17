library(bnlearn)
library(Rgraphviz)
library(gRain)
library(gridExtra)
library(tidyverse)
library(corrplot)
library(FSelector)
library(ggplot2)
library(reshape2)

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
data <- data %>% mutate(across(everything(), as.numeric))
str(data)

# Feature selection
info_gain <- information.gain(as.formula(paste("V25", "~ .")), data)

# Order features by their importance (descending)
info_gain_ordered <- info_gain[order(-info_gain$attr_importance), , drop = FALSE]

# Select top 24 features
selected_features <- rownames(info_gain_ordered)[1:7]

# Filter dataset to keep selected features + target
filtered_data <- data[, c(selected_features, "V25")]
str(filtered_data)
filtered_data <- filtered_data %>% mutate(across(everything(), factor))

# Create train/test split (80/20)
set.seed(123)
train_indices <- sample(1:nrow(filtered_data), 0.8 * nrow(filtered_data))
train <- filtered_data[train_indices, ]
test <- filtered_data[-train_indices, ]
features <- setdiff(names(filtered_data), "V25")
blacklist <- tiers2blacklist(list(features, "V25"))

# Try different structure learning approaches
models <- list(
  hc_k2 = hc(train, blacklist = blacklist, score = "k2"),
  hc_bic = hc(train, blacklist = blacklist, score = "bic"),
  hc_bde = hc(train, blacklist = blacklist, score = "bde"),
  tabu = tabu(train, blacklist = blacklist),
  iamb = cextend(iamb(train, blacklist = blacklist)),
  mmhc = mmhc(train, blacklist = blacklist),
  rsmax2 = rsmax2(train, blacklist = blacklist)
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

# Evaluate models
results <- lapply(fitted_models, evaluate_model, test)

# Print results
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

# Plot confusion matrices
plot_confusion_matrices <- function(results) {
  plots <- list()
  
  for (model_name in names(results)) {
    conf_mat <- results[[model_name]]$confusion
    conf_df <- as.data.frame(as.table(conf_mat))
    names(conf_df) <- c("Predicted", "Actual", "Frequency")
    
    accuracy <- round(results[[model_name]]$accuracy, 3)
    
    p <- ggplot(conf_df, aes(x = Actual, y = Predicted, fill = Frequency)) +
      geom_tile() +
      geom_text(aes(label = Frequency),
                color = "black",
                size = 3) +
      scale_fill_gradient(low = "white", high = "steelblue") +
      labs(
        title = paste0(toupper(model_name), " (Accuracy: ", accuracy, ")"),
        x = "Actual Class",
        y = "Predicted Class"
      ) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5),
            axis.text.x = element_text(angle = 45, hjust = 1))
    
    plots[[model_name]] <- p
  }
  
  return(plots)
}

# Generate all confusion matrix plots
confusion_plots <- plot_confusion_matrices(results)

# Arrange and display the plots
top_models <- c("hc_k2", "iamb", "hc_bde")  # Top 3 models by accuracy
grid.arrange(grobs = confusion_plots[top_models], ncol = 2)

# Display all plots
grid.arrange(grobs = confusion_plots, ncol = 2)

for (model_name in names(models)) {
  graphviz.plot(models[[model_name]], )
}