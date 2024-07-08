from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

model = Pipeline(steps=[('regressor', LinearRegression())])
df_train = pd.read_csv("./data/train_tfidf_features.csv")
X_train = df_train.drop(['label', 'id'], axis=1)
y_train = df_train['label']

df_test = pd.read_csv("./data/test_tfidf_features.csv")
X_test = df_test.drop(['id'], axis=1)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

num_ones = np.count_nonzero(y_hat)
num_zeros = len(y_hat) - num_ones
print("Number of 1s: ", num_ones)
print("Number of 0s: ", num_zeros)