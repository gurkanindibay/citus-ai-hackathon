import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from tensorflow import keras
from tensorflow.keras import layers

# Read the data from the CSV file
data = pd.read_csv("metrics_data_04.csv")

# Split the data into features (X) and target variable (y)
X = data.drop(["Time", "ShardSplit Required"], axis=1)
y = data["ShardSplit Required"]

# Perform one-hot encoding for categorical columns
categorical_cols = ["Node", "Tenant_id", "Shard"]
ct = ColumnTransformer(
    [("encoder", OneHotEncoder(), categorical_cols)], remainder="passthrough"
)
X_encoded = ct.fit_transform(X)

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a neural network model
model = keras.Sequential(
    [
        layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)

print("Accuracy:", accuracy)

# Make predictions on the test set
y_pred_prob = model.predict(X_test)
threshold = 0.49
y_pred_labels = ["Yes" if prob >= threshold else "No" for prob in y_pred_prob]

# Convert actual labels to 'Yes' or 'No'
y_test_labels = le.inverse_transform(y_test.ravel())

# Iterate in y_pred and see if there is any 'Yes' in it
result = [i for i in y_pred_labels if i == 'Yes']
print("result")
print(result)
print(f"{len(result)}/{len(y_pred_labels)}")

# Print the predicted labels and the actual labels
print("Predicted labels:")
print(y_pred_labels)
print("Actual labels:")
print(y_test_labels)
