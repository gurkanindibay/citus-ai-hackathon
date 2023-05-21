from ai_models.knn_predictor import knn_produce_data
from ai_models.knn_model import knn_create_model_file
from enum import Enum
import argparse

class BuildType(Enum):
    knn_export_data = 1
    knn_create_model = 2
    split_shards = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--action", required=False, choices=[b.name for b in BuildType])
    


    args = parser.parse_args()

    if args.action == BuildType.knn_export_data.name :
        knn_produce_data()
    elif args.action == BuildType.knn_create_model.name :
        knn_create_model_file()
    