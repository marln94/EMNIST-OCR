import pandas as pd
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv("data/emnist-balanced-train.csv")

train, validate = train_test_split(raw_data, test_size=0.1) # change this split however you want

x_train = train.values[:,1:]
y_train = train.values[:,0]

print("x_train\n")
print(x_train.shape)

print("y_train\n")
print(y_train.shape)

# x_validate = validate.values[:,1:]
# y_validate = validate.values[:,0]