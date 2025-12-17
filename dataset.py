import os
import torch
import numpy as np
import json
from PIL import Image


class LFDataset:
    def __init__(self, folder):
        self.folder = folder
        self.camera_matrix = torch.tensor(
            np.loadtxt(f"{self.folder}/camera_matrix.txt"), dtype=torch.float64
        )
        with open(f"{self.folder}/metadata.json", "r") as f:
            self.metadata = json.load(f)
        self.frames = list(sorted(
            [item for item in os.listdir(self.folder) if "LF_" in item]
        ))
        self.size = len(self.frames)
        self.camera_poses_dir = os.path.join(self.folder, "camera_poses")
        self.depth_dir = os.path.join(self.folder, "depth")
        self.object_poses_dir = os.path.join(self.folder, "object_poses")

        self.camera_poses = []
        for pose_file in sorted(os.listdir(self.camera_poses_dir)):
            if pose_file.endswith(".txt"):
                pose_path = os.path.join(self.camera_poses_dir, pose_file)
                pose = torch.tensor(np.loadtxt(pose_path), dtype=torch.float64)
                self.camera_poses.append(pose)
        self.camera_poses = torch.stack(self.camera_poses, dim=0).reshape(
            self.metadata["n_views"][0],
            self.metadata["n_views"][1],
            *self.camera_poses[0].shape,
        )

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        frame_path = os.path.join(self.folder, self.frames[idx])
        img_paths = sorted(
            [
                os.path.join(frame_path, f)
                for f in os.listdir(frame_path)
                if f.endswith(".png")
            ]
        )

        imgs = [
            torch.tensor(np.array(Image.open(p)), dtype=torch.float64)
            for p in img_paths
        ]
        LF = torch.stack(imgs, dim=0)
        LF = LF.view(
            self.metadata["n_views"][0], self.metadata["n_views"][1], *imgs[0].shape
        )
        LF /= LF.max()
        s_mid, t_mid = LF.shape[0] // 2, LF.shape[1] // 2
        depth = np.array(Image.open(os.path.join(self.depth_dir, f"{idx:04d}.png")))
        depth = torch.tensor(depth, dtype=torch.float64) / 1000.0
        object_pose = np.loadtxt(os.path.join(self.object_poses_dir, f"{idx:04d}.txt"))
        object_pose = torch.tensor(object_pose, dtype=torch.float64)

        return {
            "LF": LF.cuda(),
            "depth": depth.cuda(),
            "camera_matrix": self.camera_matrix.cuda(),
            "object_pose": object_pose.cuda(),
            "camera_poses": self.camera_poses.cuda(),
            "camera_poses_rel": (torch.linalg.inv(self.camera_poses[s_mid, t_mid]) @ self.camera_poses).cuda(),
            "baseline": self.metadata["x_spacing"],
        }

# class LFDataset:
#     def __init__(self, folder, is_ref=False):
#         self.folder = folder
#         self.is_ref = is_ref
#         self.camera_matrix = torch.tensor(
#             np.loadtxt(f"{self.folder}/camera_matrix.txt"), dtype=torch.float64
#         )
#         with open(f"{self.folder}/metadata.json", "r") as f:
#             self.metadata = json.load(f)
#         if self.is_ref:
#             self.folder = f"{self.folder}/ref_views"
#             self.target_pose = torch.tensor(
#                 np.loadtxt(f"{self.folder}/obj_pose.txt"), dtype=torch.float64
#             )
#         self.frames = sorted(
#             [item for item in os.listdir(self.folder) if "LF_" in item]
#         )
#         self.size = len(self.frames)

#     def __len__(self):
#         return self.size

#     def __getitem__(self, idx):
#         frame_path = os.path.join(self.folder, self.frames[idx])
#         img_dir = os.path.join(frame_path, "imgs")
#         img_paths = sorted(
#             [
#                 os.path.join(img_dir, f)
#                 for f in os.listdir(img_dir)
#                 if f.endswith(".png")
#             ]
#         )

#         imgs = [
#             torch.tensor(np.array(Image.open(p)), dtype=torch.float64)
#             for p in img_paths
#         ]
#         LF = torch.stack(imgs, dim=0)
#         LF = LF.view(
#             self.metadata["n_views"][0], self.metadata["n_views"][1], *imgs[0].shape
#         )
#         LF /= LF.max()

#         pose_dir = os.path.join(frame_path, "poses")
#         pose_paths = sorted(
#             [
#                 os.path.join(pose_dir, f)
#                 for f in os.listdir(pose_dir)
#                 if f.endswith(".txt")
#             ]
#         )
#         poses = [torch.tensor(np.loadtxt(p), dtype=torch.float64) for p in pose_paths]
#         poses = torch.stack(poses, dim=0).view(
#             self.metadata["n_views"][0], self.metadata["n_views"][1], *poses[0].shape
#         )

#         masks, depths = None, None

#         if os.path.exists(os.path.join(frame_path, "depth")):
#             depth_dir = os.path.join(frame_path, "depth")
#             depth_paths = sorted(
#                 [
#                     os.path.join(depth_dir, f)
#                     for f in os.listdir(depth_dir)
#                     if f.endswith(".npy")
#                 ]
#             )
#             depths = [np.load(depth_path) / 1000.0 for depth_path in depth_paths]
#             depths = [torch.tensor(d, dtype=torch.float64) for d in depths]
#             depths = torch.stack(depths, dim=0).view(
#                 self.metadata["n_views"][0],
#                 self.metadata["n_views"][1],
#                 *depths[0].shape,
#             )

#         if os.path.exists(os.path.join(frame_path, "masks")):
#             mask_dir = os.path.join(frame_path, "masks")
#             mask_paths = sorted(
#                 [
#                     os.path.join(mask_dir, f)
#                     for f in os.listdir(mask_dir)
#                     if f.endswith(".png")
#                 ]
#             )
#             masks = [
#                 torch.tensor(np.array(Image.open(p)), dtype=torch.float64)
#                 for p in mask_paths
#             ]
#             masks = torch.stack(masks, dim=0).view(
#                 self.metadata["n_views"][0],
#                 self.metadata["n_views"][1],
#                 *masks[0].shape,
#             )
#             masks /= masks.max()
#         s_mid, t_mid = LF.shape[0] // 2, LF.shape[1] // 2
#         result = {
#             "frame_path": frame_path,
#             "LF": LF.cuda(),
#             "camera_matrix": self.camera_matrix.cuda(),
#             "camera_poses": poses.cuda(),
#             "camera_poses_rel": (torch.linalg.inv(poses[s_mid, t_mid]) @ poses).cuda(),
#             "masks": masks.cuda() if masks is not None else None,
#             "depths": depths.cuda() if depths is not None else None,
#             "baseline": self.metadata["x_spacing"],
#             "target_pose": torch.tensor(
#                 np.loadtxt(
#                     f"{self.folder}/obj_pose.txt"
#                     if self.is_ref
#                     else f"{frame_path}/obj_pose.txt"
#                 ),
#                 dtype=torch.float64,
#             ).cuda(),
#         }

#         if os.path.exists(os.path.join(frame_path, "DAM_depth.npy")):
#             dam_disparity = torch.tensor(
#                 np.load(os.path.join(frame_path, "DAM_depth.npy")),
#                 dtype=torch.float64,
#             ).cuda()
#             result["dam_disparity"] = dam_disparity

#         if os.path.exists(os.path.join(frame_path, "object_pose.txt")):
#             object_pose = torch.tensor(
#                 np.loadtxt(os.path.join(frame_path, "object_pose.txt")),
#                 dtype=torch.float64,
#             ).cuda()
#             result["object_pose"] = object_pose

#         obj_pose_path = (
#             os.path.join(self.folder, "object_pose.txt")
#             if self.is_ref
#             else os.path.join(frame_path, "object_pose.txt")
#         )
#         if os.path.exists(obj_pose_path):
#             result["object_pose"] = torch.tensor(
#                 np.loadtxt(obj_pose_path), dtype=torch.float64
#             ).cuda()

#         return result


if __name__ == "__main__":
    dataset = LFDataset("data/jug_tilt_prod")
    result = dataset[0]
    for key, value in result.items():
        if not type(value) is str:
            print(key)
            # print(value)
