from dataset import LFDataset
import torch
import yaml
import os

from pose_tracker import PoseTracker
from metrics import compute_pose_errors
import shutil
from utilities import visualize_tracking
from tqdm import tqdm
from scipy.spatial import cKDTree
import numpy as np
from sklearn.metrics import auc
import trimesh
from tqdm import tqdm


with open("config.yaml", "r") as f:
    EXP_CONFIG = yaml.safe_load(f)


def transform_pts(pts, tf):
    """Transform 2d or 3d points
    @pts: (...,N_pts,3)
    @tf: (...,4,4)
    """
    if len(tf.shape) >= 3 and tf.shape[-3] != pts.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]


def add_err(pred, gt, model_pts, symetry_tfs=np.eye(4)[None]):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).
    """
    pred_pts = transform_pts(model_pts, pred)
    gt_pts = transform_pts(model_pts, gt)
    e = np.linalg.norm(pred_pts - gt_pts, axis=-1).mean()
    return e


def adds_err(pred, gt, model_pts):
    """
    @pred: 4x4 mat
    @gt:
    @model: (N,3)
    """
    pred_pts = transform_pts(model_pts, pred)
    gt_pts = transform_pts(model_pts, gt)
    nn_index = cKDTree(pred_pts)
    nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
    e = nn_dists.mean()
    return e


def get_metrics(gt_poses, mesh_path, estimated_poses, threshold_max=0.1):
    thresholds_space = np.linspace(0, threshold_max, 100)
    mesh = trimesh.load(mesh_path)
    mesh.apply_scale(EXP_CONFIG.get("upscale-model", 1.0))
    gt_pc = mesh.vertices.copy()
    adds_vals = []
    add_vals = []
    for i in range(len(gt_poses)):
        object_to_cam = gt_poses[i].cpu().numpy()
        estimated_pose = estimated_poses[i].cpu().numpy()
        add_val = add_err(estimated_pose, object_to_cam, gt_pc)
        adds_val = adds_err(estimated_pose, object_to_cam, gt_pc)
        add_vals.append(add_val)
        adds_vals.append(adds_val)
    adds_vals = np.array(adds_vals)
    add_vals = np.array(add_vals)
    adds_accuracies = [(adds_vals < t).mean() for t in thresholds_space]
    add_accuracies = [(add_vals < t).mean() for t in thresholds_space]
    adds_auc = auc(np.linspace(0, 1, 100), adds_accuracies)
    add_auc = auc(np.linspace(0, 1, 100), add_accuracies)
    return adds_vals, add_vals, adds_auc, add_auc


def main():
    # PREPARE FOLDER
    exp_name = EXP_CONFIG["exp-name"]
    exp_path = f"experiments/{exp_name}"
    os.makedirs(exp_path, exist_ok=True)
    shutil.copy("config.yaml", f"{exp_path}/config.yaml")
    is_synth = False

    dataset_path = EXP_CONFIG["dataset-path"]
    dataset = LFDataset(dataset_path, is_ref=False)
    # dataset = LFSynthData(dataset_path)
    i_start = 10
    if_finish = -1
    frame0 = dataset[i_start]
    camera_matrix = frame0["camera_matrix"]
    obj_to_base_pose = frame0["object_pose"]
    camera_poses = frame0["camera_poses"]
    s_mid, t_mid = camera_poses.shape[0] // 2, camera_poses.shape[1] // 2
    cam_to_base = camera_poses[s_mid, t_mid]
    init_pose = torch.linalg.inv(cam_to_base) @ obj_to_base_pose
    obj_to_cam = torch.linalg.inv(cam_to_base) @ obj_to_base_pose

    gt_poses = [
        obj_to_cam,
    ]
    tracker = PoseTracker(
        dataset[i_start], enable_cache=EXP_CONFIG.get("enable-cache", False)
    )
    tracker.poses = [gt_poses[0].float()]
    coarse_poses = [gt_poses[0].float()]
    for i in tqdm(range(i_start + 1, len(dataset))):
        frame = dataset[i]
        _, coarse_pose = tracker.track_pose(frame, i)
        coarse_poses.append(coarse_pose)
        gt_poses.append(
            torch.linalg.inv(frame["camera_poses"][s_mid, t_mid]) @ frame["object_pose"]
        )
        if i == if_finish:
            break

        # EVALUATE
        gt_poses_stacked = torch.stack(gt_poses).float().detach()
        est_poses_stacked = torch.stack(tracker.poses).float().detach()
        coarse_poses_stacked = torch.stack(coarse_poses).float().detach()

        pose_errors = compute_pose_errors(gt_poses_stacked, est_poses_stacked)
        _, _, adds_err_vals, add_err_vals = get_metrics(
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
            i_start,
            est_poses_stacked,
            camera_matrix,
            f"{exp_path}/vis",
            synth=is_synth,
        )
        visualize_tracking(
            dataset_path,
            i_start,
            coarse_poses_stacked,
            camera_matrix,
            f"{exp_path}/vis_coarse",
            synth=is_synth,
        )


if __name__ == "__main__":
    main()
