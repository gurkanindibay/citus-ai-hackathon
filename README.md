We will demo in 3 phase

1. Create model file (ai_models/models/knn_model.pkl)
    python main_executor.py --action knn_create_model
2. Get the shard_split decisions from the input
    python main_executor.py --action knn_export_data
3. Shard split using the data created in step 2
    python main_executor.py --action split_shards (missing now)
