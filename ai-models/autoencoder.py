import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create the dataset
data = pd.read_csv('sample_data.csv')

# Select the columns for training
selected_columns = ['Node', 'Tenant_id', 'Shard', 'Query Rate (QPS)', 'Data Growth Rate (GB/day)',
                    'Disk Space Utilization (%)', 'CPU Utilization (%)', 'Memory Utilization (%)',
                    'Shard Size (GB)', 'Query Response Time (ms)']

# Scale the selected columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[selected_columns])

# Create the autoencoder model
input_dim = len(selected_columns)
encoding_dim = 5  # Dimension of the encoded representation
autoencoder = Sequential()
autoencoder.add(Dense(8, activation='relu', input_dim=input_dim))
autoencoder.add(Dense(encoding_dim, activation='relu'))
autoencoder.add(Dense(8, activation='relu'))
autoencoder.add(Dense(input_dim, activation='linear'))

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=16, shuffle=True)

# Obtain the encoded representations of the data
encoder = Sequential(autoencoder.layers[:2])  # Extract the encoder part of the autoencoder
encoded_data = encoder.predict(scaled_data)

# Print the encoded representations
print(encoded_data)
