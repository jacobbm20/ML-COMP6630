# Re-import necessary libraries after reset
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Reload the Excel file and data
train_file = "train.csv"
test_file = "test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Prepare features and target
X_train = train_df.drop(columns=["House ID", "Construction type"])
y_train = train_df["Construction type"]

X_test = test_df.drop(columns=["House ID", "Construction type"])
y_test = test_df["Construction type"]

# Encode the target labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# 1. Train classifier with default parameters
clf_default = DecisionTreeClassifier(random_state=42)
clf_default.fit(X_train, y_train_encoded)

# Compute training and test accuracy
train_acc_default = accuracy_score(y_train_encoded, clf_default.predict(X_train))
test_acc_default = accuracy_score(y_test_encoded, clf_default.predict(X_test))

# 2. Try different max depths and record test accuracies
depth_results = {}
for depth in range(1, 11):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train_encoded)
    acc = accuracy_score(y_test_encoded, clf.predict(X_test))
    depth_results[depth] = acc

# 4. Inference on custom test point
sample = pd.DataFrame([{
    "Local Price": 9.0384,
    "Bathrooms": 1,
    "Land Area": 7.8,
    "Living area": 1.5,
    "# Garages": 1.5,
    "# Rooms": 7,
    "# Bedrooms": 3,
    "Age of home": 23
}])

predicted_class = le.inverse_transform(clf_default.predict(sample))[0]

print(train_acc_default, test_acc_default, depth_results, predicted_class)

1.0 0.4 {1: 0.4, 2: 0.8, 3: 0.4, 4: 0.4, 5: 
         0.4, 6: 0.4, 7: 0.4, 8: 0.4, 9: 0.4, 10: 0.4} 

Apartment