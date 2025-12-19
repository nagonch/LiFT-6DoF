from dataset import LFDataset
from disparity import get_frame_disparity
from functional import get_gs_initialization
import torch
from utilities import Visualizer, backproject_depth_to_pointcloud
from tqdm import tqdm
from gs_train import train_splat
from refine_pose import refine_splat_camera_poses
import os
from icp_align import align_colored_point_clouds_np
import yaml

with open("config.yaml", "r") as f:
    EXP_CONFIG = yaml.safe_load(f)


def align_point_clouds(pc_src, pc_tgt, colors_src, colors_tgt):
    pc_src = torch.cat((pc_src, colors_src), dim=1)
    pc_tgt = torch.cat((pc_tgt, colors_tgt), dim=1)
    pose_rel, _ = align_colored_point_clouds_np(
        pc_src.cpu().numpy(), pc_tgt.cpu().numpy()
    )
    pose_rel = torch.tensor(pose_rel).cuda().float()
    return pose_rel


class PoseTracker:
    def __init__(
        self,
        init_frame,
        init_opacity=EXP_CONFIG["pose-tracker"]["init_opacity"],
        enable_viewer=EXP_CONFIG["pose-tracker"]["enable_viewer"],
        n_develop_epochs=EXP_CONFIG["pose-tracker"]["n_develop_epochs"],
        develop_lr=EXP_CONFIG["pose-tracker"]["develop_lr"],
        max_gaussians=EXP_CONFIG["pose-tracker"]["max_gaussians"],
        n_pose_refine_epochs=EXP_CONFIG["pose-tracker"]["n_pose_refine_epochs"],
        refine_lr=EXP_CONFIG["pose-tracker"]["refine_lr"],
        enable_cache=EXP_CONFIG["pose-tracker"]["enable_cache"],
    ):
        self.camera_matrix = init_frame["camera_matrix"]
        self.init_opacity = init_opacity
        self.enable_viewer = enable_viewer
        self.n_develop_epochs = n_develop_epochs
        self.develop_lr = develop_lr
        self.max_gaussians = max_gaussians
        self.n_pose_refine_epochs = n_pose_refine_epochs
        self.enable_cache = enable_cache
        self.refine_lr = refine_lr

        if (
            enable_cache
            and os.path.exists(f'{init_frame["frame_path"]}/splat.pt')
            and os.path.exists(f'{init_frame["frame_path"]}/pc.pt')
        ):
            self.pc = torch.load(f'{init_frame["frame_path"]}/pc.pt')
            self.splat = torch.load(f'{init_frame["frame_path"]}/splat.pt')
            self.train_images, self.train_poses, self.train_masks = self.get_train_data(
                init_frame
            )
        else:
            self.pc, means, sh_coeffs, scales = get_gs_initialization(init_frame)
            self.splat, self.train_images, self.train_poses, self.train_masks = (
                self.develop_splat(init_frame, means, sh_coeffs, scales)
            )
        torch.save(self.pc, f'{init_frame["frame_path"]}/pc.pt')
        torch.save(self.splat, f'{init_frame["frame_path"]}/splat.pt')
        self.camera_0_to_base = init_frame["camera_poses"][
            init_frame["camera_poses"].shape[0] // 2,
            init_frame["camera_poses"].shape[1] // 2,
        ]
        self.poses = [self.initialise_pose(self.pc)]  # object_to_cam_0
        self.poses_gt = [self.get_gt_pose(init_frame)]  # object_to_cam_0

    def get_gt_pose(self, frame):
        obj_to_base_pose = frame["object_pose"]
        obj_to_cam = torch.linalg.inv(self.camera_0_to_base) @ obj_to_base_pose
        return obj_to_cam.float()

    def get_pc(self, frame):
        disparity = get_frame_disparity(frame)
        depth = frame["camera_matrix"][0, 0] * frame["baseline"] / (disparity + 1e-8)
        mask = frame["masks"][4, 4]
        depth = depth * (mask > 0).float()
        points = backproject_depth_to_pointcloud(None, depth, frame["camera_matrix"])
        points = points[mask.reshape(-1) > 0]
        return points

    def initialise_pose(self, means_init):
        init_pose = torch.eye(4).cuda().float()
        init_pose[:3, 3] = means_init.mean(dim=0)
        return init_pose.float()

    def get_train_data(self, frame):
        LF = frame["LF"] * frame["masks"][..., None]
        masks = frame["masks"].reshape(-1, LF.shape[2], LF.shape[3])
        images = LF.reshape(-1, LF.shape[2], LF.shape[3], 3)
        images[masks == 0] = 1
        poses = frame["camera_poses_rel"].reshape(-1, 4, 4)
        train_images, train_poses, train_masks = (
            images.float().cuda(),
            poses.float().cuda(),
            masks.float().cuda(),
        )
        return train_images, train_poses, train_masks

    def develop_splat(self, frame, means_init, colors_init, scales_init):
        train_images, train_poses, train_masks = self.get_train_data(frame)
        opacities = torch.logit(torch.full((means_init.shape[0],), self.init_opacity))
        quats = torch.randn(opacities.shape[0], 4).float().cuda()
        gs_train_data = {
            "images": train_images,
            "poses": train_poses,
            "means": means_init.float().cuda(),
            "masks": train_masks,
            "colors": colors_init.float().cuda(),
            "scales": scales_init.float().cuda(),
            "camera_matrix": self.camera_matrix.float().cuda(),
            "opacities": opacities.float().cuda(),
            "quats": quats.float().cuda(),
        }
        splat = train_splat(
            gs_train_data,
            verbose=False,
            enable_viewer=self.enable_viewer,
            n_epochs=self.n_develop_epochs,
            lr=self.develop_lr,
        )
        return splat, train_images, train_poses, train_masks

    def track_pose(self, frame, i):
        self.poses_gt.append(self.get_gt_pose(frame))  # object_to_cam_0
        # Init pose estimation
        if (
            self.enable_cache
            and os.path.exists(f'{frame["frame_path"]}/splat.pt')
            and os.path.exists(f'{frame["frame_path"]}/pc.pt')
        ):
            pc = torch.load(f'{frame["frame_path"]}/pc.pt')
            splat = torch.load(f'{frame["frame_path"]}/splat.pt')
            train_images, train_poses, train_masks = self.get_train_data(frame)
        else:
            pc, means, sh_coeffs, scales = get_gs_initialization(frame)
            # Pose refinement via splats
            splat, train_images, train_poses, train_masks = self.develop_splat(
                frame, means, sh_coeffs, scales
            )
            torch.save(pc, f'{frame["frame_path"]}/pc.pt')
            torch.save(splat, f'{frame["frame_path"]}/splat.pt')
        last_pose = self.poses[-1].cuda()  # object_prev_to_cam0
        pose_rel = align_point_clouds(
            self.splat["means"],
            splat["means"],
            self.splat["colors"][:, 0],
            splat["colors"][:, 0],
        )  # pose_rel @ points_prev_to_cam0 = points_to_cam0
        pose_rel_gt = self.poses_gt[-1] @ torch.linalg.inv(
            self.poses_gt[-2]
        )  # object_to_cam0 @ inv(object_to_cam0_prev)
        # torch.save(self.poses_gt[-2], f"pose_prev_gt.pt")
        # torch.save(self.poses_gt[-1], f"pose_gt.pt")
        # torch.save(self.pc, f"pc_prev.pt")
        # torch.save(pc, f"pc_curr.pt")
        # raise
        coarse_pose = pose_rel @ last_pose
        pose_rel = refine_splat_camera_poses(
            self.splat,
            splat,
            self.train_images.to("cuda", non_blocking=True),
            self.train_poses.to("cuda", non_blocking=True),
            self.train_masks.to("cuda", non_blocking=True),
            train_images.to("cuda", non_blocking=True),
            train_poses.to("cuda", non_blocking=True),
            train_masks.to("cuda", non_blocking=True),
            pose_rel,
            pose_rel_gt,
            last_pose,
            camera_matrix=self.camera_matrix.float().cuda(),
            verbose=True,
            enable_viewer=self.enable_viewer,
            n_epochs=self.n_pose_refine_epochs,
            lr=self.refine_lr,
        )

        pose = pose_rel @ last_pose
        self.poses.append(pose)
        self.pc = torch.clone(pc)
        self.train_images = train_images
        self.train_poses = train_poses
        self.train_masks = train_masks
        self.splat = splat
        del splat
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return pose, coarse_pose


if __name__ == "__main__":
    pass
