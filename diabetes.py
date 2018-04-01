from sklearn import datasets
from sklearn import linear_model

ds = datasets.load_diabetes()

train_features,train_output = ds.data,ds.target

lss = linear_model.Lasso(alpha = 2)

lss.fit(train_features,train_output)

print(lss.coef_)