# X, y = load_iris(return_X_y=True)
# clf = LogisticRegression(random_state=0).fit(X, y)
# clf.predict(X[:2, :])
# def manual_fold_training(
#         clf_model : LogisticRegression,
#         data : pl.DataFrame,
#         fold_number : int = 0,
#         *,
#         target_column : str = "is_anomaly"
# ):
#     scores = {"fold":fold_number}
#
#     fold_train = data.filter(pl.col("fold") != fold_number).drop("fold")
#     fold_test = data.filter(pl.col("fold") == fold_number).drop("fold")
#     X_train = (fold_train.select(pl.exclude(target_column))
#                .select(pl.col(pl.Float64)).with_columns(
#         pl.all().fill_nan(0)
#     ).to_numpy())
#     y_train =  fold_train.select(target_column).to_series().to_list()
#     clf_model.fit(X_train,y_train)
#     X_test = fold_test.select(pl.exclude(target_column)).to_numpy()
#     y_test =  fold_test.select(pl.col(target_column)).to_series().to_numpy()
#     y_pred = clf_model.predict(X_test)
#     scores["f1_score"] = f1_score(y_test,y_pred)
#     scores["recall"] = recall_score(y_test,y_pred)
#
#     return scores
#
#
# logistic_regressor = LogisticRegression(solver='liblinear',C=100.0,penalty="l2",max_iter=1000)
#
# fold_scores = []
# for fold in train["fold"].unique().to_list():
#     fold_scores.append(manual_fold_training(clf_model=logistic_regressor,
#                                             data=train,
#                                             fold_number=fold))
#
#
# df_scores = pl.DataFrame(fold_scores)
# mean_row = df_scores.select(pl.all().mean().cast(pl.Float64))
# df_with_mean = df_scores.with_columns(pl.col("fold").cast(pl.Utf8)).vstack(mean_row)
# df_with_mean
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np


def EnsembleFilter(X, y, k=5, voting='consensus'):
    """
    Apply the Ensemble Filter to detect and remove noisy examples from the training data.

    Parameters:
    - X: numpy array of shape (n_samples, n_features), the training data.
    - y: numpy array of shape (n_samples,), the training labels.
    - k: int, number of folds for cross-validation.
    - voting: str, either 'consensus' or 'majority' voting scheme.

    Returns:
    - clean_X: numpy array of shape (n_clean_samples, n_features), the filtered training data.
    - clean_y: numpy array of shape (n_clean_samples,), the filtered training labels.
    """

    # Initialize classifiers
    classifiers = [
        DecisionTreeClassifier(),  # C4.5 equivalent
        KNeighborsClassifier(n_neighbors=1),  # 1-NN
        LinearDiscriminantAnalysis()  # LDA
    ]

    # Perform k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    mislabel_counts = np.zeros(len(y))

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train each classifier and make predictions
        for clf in classifiers:
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            mislabel_counts[test_index] += (predictions != y_test)  # Increment mislabel count for misclassified samples

    # Determine which examples to keep based on voting scheme
    if voting == 'consensus':
        # Consensus voting: retain examples misclassified by all classifiers
        noisy_indices = np.where(mislabel_counts == len(classifiers))[0]
    elif voting == 'majority':
        # Majority voting: retain examples misclassified by more than half of classifiers
        noisy_indices = np.where(mislabel_counts > len(classifiers) / 2)[0]
    else:
        raise ValueError("Voting must be either 'consensus' or 'majority'")

    # Filter out noisy examples
    clean_indices = np.setdiff1d(np.arange(len(y)), noisy_indices)
    clean_X = X[clean_indices]
    clean_y = y[clean_indices]

    return clean_X, clean_y,noisy_indices
