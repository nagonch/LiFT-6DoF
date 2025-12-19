import torch
import yaml
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.metrics import auc

with open("config.yaml", "r") as f:
    EXP_CONFIG = yaml.safe_load(f)


def rotation_angle_deg(R_err):
    trace = R_err.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1) / 2).clamp(-1.0, 1.0)
    return torch.acos(cos_theta) * (180.0 / torch.pi)


def compute_pose_errors(gt_poses, est_poses):
    assert gt_poses.shape == est_poses.shape
    assert gt_poses.shape[-2:] == (4, 4)

    N = gt_poses.shape[0]

    # Absolute errors
    R_gt = gt_poses[:, :3, :3]
    t_gt = gt_poses[:, :3, 3]
    R_est = est_poses[:, :3, :3]
    t_est = est_poses[:, :3, 3]

    R_err_abs = R_est @ R_gt.transpose(-1, -2)  # (N, 3, 3)
    rot_err_abs = rotation_angle_deg(R_err_abs)  # (N,)
    trans_err_abs = torch.norm(t_est - t_gt, dim=1)  # (N,)

    # Relative errors
    rel_rot_errs = []
    rel_trans_errs = []
    for i in range(N - 1):
        T_gt_rel = torch.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        T_est_rel = torch.linalg.inv(est_poses[i]) @ est_poses[i + 1]

        R_gt_rel = T_gt_rel[:3, :3]
        R_est_rel = T_est_rel[:3, :3]
        t_gt_rel = T_gt_rel[:3, 3]
        t_est_rel = T_est_rel[:3, 3]

        R_err_rel = R_est_rel @ R_gt_rel.transpose(-1, -2)
        rel_rot_errs.append(rotation_angle_deg(R_err_rel.unsqueeze(0)))
        rel_trans_errs.append(torch.norm(t_est_rel - t_gt_rel).unsqueeze(0))

    rel_rot_errs = torch.cat(rel_rot_errs)
    rel_trans_errs = torch.cat(rel_trans_errs)

    return {
        "mean_abs_rot_deg": rot_err_abs.mean().item(),
        "mean_abs_trans": trans_err_abs.mean().item(),
        "mean_rel_rot_deg": rel_rot_errs.mean().item(),
        "mean_rel_trans": rel_trans_errs.mean().item(),
    }


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


def get_add_metrics(gt_poses, mesh_path, estimated_poses, threshold_max=0.1):
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


if __name__ == "__main__":
    pass
