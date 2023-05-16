import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('input_data.csv')

data = data.drop(['Time'], axis=1)

loaded_model = joblib.load('knn_model.pkl')

# Make predictions on the test set
y_pred = loaded_model.predict(data)

# Print the predictions
print(y_pred)
