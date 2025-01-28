library(mRMRe)
file.name  <- paste0("../data/train", ".csv")
df <- read.csv(file.name, header = TRUE)
df$fold <- NULL
df$is_anomaly <- as.numeric(df$is_anomaly)
file.data <- mRMR.data(data = data.frame(df))
results <- mRMR.classic("mRMRe.Filter", data = file.data, target_indices = 76, feature_count = 35)
solutions(results)
