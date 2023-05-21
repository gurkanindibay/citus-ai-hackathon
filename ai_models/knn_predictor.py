import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib


def knn_produce_data():
    data_dir = 'ai_models/data'
    model_dir = 'ai_models/models'

    # Validate the model

    print("Loading data")
    input_data = pd.read_csv(f'{data_dir}/metrics_data_blank_01.csv')
    # Split the data into features (X) and target variable (y)
    X_data = input_data.drop(['Time', 'ShardSplit Required'], axis=1)

    print("Scaling data")
    scaler = StandardScaler()
    X_data_scaled = scaler.fit_transform(X_data)


    print("Loading model")
    loaded_model = joblib.load(f'{model_dir}/knn_model.pkl')

    print("Predicting...")
    y_pred = loaded_model.predict(X_data_scaled)

    X_data['ShardSplit Required'] = y_pred

    # Iterate in X_data and see if there is any 'Yes' in it
    split_required_data = X_data[X_data['ShardSplit Required'] == 'Yes']


    data_to_export=split_required_data.loc[:,['Shard','Tenant_id','ShardSplit Required']]
    
    data_to_export.rename(columns={'Shard': 'shardid', 'Tenant_id': 'tenant', 'ShardSplit Required': 'decision'}, inplace=True)
    
    data_to_export['tablename']='stats'
    print("Exporting data")
    data_to_export.to_csv(f'{data_dir}/shard_split_required.csv', index=False)

    # print("result test")
    # print(result_test)
    # print(f"{len(result_pred)}/{len(y_pred)}")

    # print("Result prediction:", result_pred)

