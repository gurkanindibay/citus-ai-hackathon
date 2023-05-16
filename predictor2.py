import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Read the data from the CSV file
data = pd.read_csv('sample_data.csv')

# Split the data into features (X) and target variable (y)
X = data.drop(['Time', 'Shard Split Needed?'], axis=1)
y = data['Shard Split Needed?']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

joblib.dump(knn, 'knn_model.pkl')

loaded_model = joblib.load('knn_model.pkl')

input_data = pd.read_csv('input_data.csv')

input_data = input_data.drop(['Time'], axis=1)

# Make predictions on the test set
y_pred = loaded_model.predict(input_data)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Print the predictions
print(y_pred)
