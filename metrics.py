import torch


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


if __name__ == "__main__":
    pass
