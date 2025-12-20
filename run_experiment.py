from src.dataset import LFDataset
import torch
import yaml
import os
from src.pose_tracker import PoseTracker
from src.metrics import compute_pose_errors, get_add_metrics
import shutil
from src.utilities import visualize_tracking
from tqdm import tqdm


with open("config.yaml", "r") as f:
    EXP_CONFIG = yaml.safe_load(f)


def main():
    # PREPARE FOLDER
    exp_name = EXP_CONFIG["exp-name"]
    exp_path = f"experiments/{exp_name}"
    os.makedirs(exp_path, exist_ok=True)
    shutil.copy("config.yaml", f"{exp_path}/config.yaml")

    dataset_path = EXP_CONFIG["dataset-path"]
    dataset = LFDataset(dataset_path)
    frame0 = dataset[0]
    camera_matrix = frame0["camera_matrix"]
    obj_to_base_pose = frame0["object_pose"]
    camera_poses = frame0["camera_poses"]
    s_mid, t_mid = camera_poses.shape[0] // 2, camera_poses.shape[1] // 2
    cam_to_base = camera_poses[s_mid, t_mid]
    obj_to_cam = torch.linalg.inv(cam_to_base) @ obj_to_base_pose

    gt_poses = [
        obj_to_cam,
    ]
    tracker = PoseTracker(frame0, enable_cache=EXP_CONFIG.get("enable-cache", False))
    tracker.poses = [gt_poses[0].float()]
    coarse_poses = [gt_poses[0].float()]
    for i in tqdm(range(len(dataset))):
        frame = dataset[i]
        _, coarse_pose = tracker.track_pose(frame, i)
        coarse_poses.append(coarse_pose)
        gt_poses.append(
            torch.linalg.inv(frame["camera_poses"][s_mid, t_mid]) @ frame["object_pose"]
        )

        # EVALUATE
        gt_poses_stacked = torch.stack(gt_poses).float().detach()
        est_poses_stacked = torch.stack(tracker.poses).float().detach()
        coarse_poses_stacked = torch.stack(coarse_poses).float().detach()

        pose_errors = compute_pose_errors(gt_poses_stacked, est_poses_stacked)
        if EXP_CONFIG["object-mesh-path"]:
            _, _, adds_err_vals, add_err_vals = get_add_metrics(
                gt_poses_stacked, EXP_CONFIG["object-mesh-path"], est_poses_stacked
            )
            pose_errors["adds_err"] = adds_err_vals
            pose_errors["add_err"] = add_err_vals

        # SAVE RESULTS
        torch.save(gt_poses_stacked, f"{exp_path}/gt_poses.pt")
        torch.save(est_poses_stacked, f"{exp_path}/est_poses.pt")
        with open(f"{exp_path}/metrics.yaml", "w") as file:
            yaml.dump(pose_errors, file, sort_keys=False)
        visualize_tracking(
            dataset_path,
            est_poses_stacked,
            camera_matrix,
            f"{exp_path}/vis",
        )
        visualize_tracking(
            dataset_path,
            coarse_poses_stacked,
            camera_matrix,
            f"{exp_path}/vis_coarse",
        )


if __name__ == "__main__":
    main()
