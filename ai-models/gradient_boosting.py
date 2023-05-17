import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Read the data from the CSV file
data = pd.read_csv('metrics_data_03.csv')

# Split the data into features (X) and target variable (y)
X = data.drop(['Time', 'ShardSplit Required'], axis=1)
y = data['ShardSplit Required']


# from sklearn.feature_selection import SelectKBest, f_classif

# k = 3 # Number of top features to select
# selector = SelectKBest(score_func=f_classif, k=k)
# X_selected = selector.fit_transform(X, y)
# selected_feature_indices = selector.get_support(indices=True)
# selected_features = X.columns[selected_feature_indices]
# print("Selected features:")
# print(selected_features)
# print("X_selected:")
# print(X_selected)

# X = X[selected_features]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import GradientBoostingClassifier

# Create and train a Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_classifier.predict(X_test)

# Calculate the accuracy of the Gradient Boosting classifier
accuracy_gb = accuracy_score(y_test, y_pred)
print('Gradient Boosting Accuracy:', accuracy_gb)


print(len(X_test))

print("y_test")
print(y_test)

print("y_pred")
print(y_pred)

# Iterate in y_pred and see if there is any 'Yes' in it
result = [i for i in y_pred if i == 'Yes']
print("result prediction")
print(result)
print(f"{len(result)}/{len(y_pred)}")

# Iterate in y_pred and see if there is any 'Yes' in it
result_test = [i for i in y_test if i == 'Yes']
print("result test")
print(result_test)
print(f"{len(result_test)}/{len(y_test)}")


# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Print the predictions
print(y_pred)
