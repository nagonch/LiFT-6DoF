from dataset import LFDataset
from depth_anything_functions import predict_depth_for_dataset
from SAM_functions import get_dataset_masks
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    args.add_argument(
        "--object_prompt",
        type=str,
        help="text description of the object",
    )
    args = args.parse_args()
    data_path = args.dataset_path
    object_prompt = args.object_prompt
    if object_prompt is None:
        with open(f"{data_path}/gdino_prompt.txt", "r") as f:
            object_prompt = f.read()

    predict_depth_for_dataset(data_path)
    get_dataset_masks(data_path, object_prompt)
