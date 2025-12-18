import torch
import sys
import os
from PIL import Image
import numpy as np

sys.path.append("Video-Depth-Anything")
from video_depth_anything.video_depth import VideoDepthAnything


def get_configs():
    return {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }


def get_video_model(encoder="vitl"):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = VideoDepthAnything(**get_configs()[encoder])
    model.load_state_dict(
        torch.load(
            f"Video-Depth-Anything/checkpoints/video_depth_anything_{encoder}.pth",
            map_location="cpu",
        )
    )
    model = model.to(device).eval()
    return model


def predict_depth_for_dataset(dataset_path):
    LF_paths = [
        f"{dataset_path}/{fname}"
        for fname in sorted(os.listdir(dataset_path))
        if "LF_" in fname
    ]
    LF_imgs_middle = list(
        sorted(
            [
                p + "/" + sorted(os.listdir(p))[len(sorted(os.listdir(p))) // 2]
                for p in LF_paths
            ]
        )
    )
    imgs = [Image.open(p) for p in LF_imgs_middle]
    imgs = np.stack(imgs, axis=0)
    model = get_video_model("vitl")
    depths = []
    depths, _ = model.infer_video_depth(
        imgs,
        -1,
    )
    for depth, LF_path in zip(depths, LF_paths):
        np.save(f"{LF_path}/predicted_depth.npy", depth)


if __name__ == "__main__":
    predict_depth_for_dataset("data/jug_tilt_prod")
