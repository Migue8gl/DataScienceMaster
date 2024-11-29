from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pca = PCA(n_components=20)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

svc = SVC(kernel='rbf', C=1, gamma='scale')
svc.fit(X_train, y_train)

cv_scores = cross_val_score(svc, X_test, y_test, cv=5)
print(f'Cross-validation accuracy: {np.mean(cv_scores):.2f}')

cinco = np.array([

    [ 0,  9, 12, 12, 12, 12,  7,  0],

    [ 0, 11,  0,  0,  0,  0,  0,  0],

    [ 0, 11, 12, 12, 11,  9,  0,  0],

    [ 0,  0,  0,  0,  0, 13,  3,  0],

    [ 0,  0,  0,  0,  0, 11,  6,  0],

    [ 0,  0,  0,  0,  0, 10,  7,  0],

    [ 0,  7,  9,  8, 11, 12,  2,  0],

    [ 0,  2, 10, 12, 10,  2,  0,  0]

])


def print_digit(digit_array):

    for row in digit_array:

        print("".join(['#' if val > 8 else '.' if val > 0 else ' ' for val in row]))


print_digit(cinco)

cinco_reshaped = cinco.flatten().reshape(1, -1) 
custom_digit_pca = pca.transform(cinco_reshaped)
predicted_digit = svc.predict(custom_digit_pca)
print(f'Predicted digit: {predicted_digit[0]}')

