import torch
from gsplat.cuda._torch_impl import _eval_sh_bases_fast
from e3nn import o3


def rgb_to_sh(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def batched_sh_least_squares(SH, C, lam=1e-4):
    B, N, D = SH.shape
    _, _, C_ = C.shape
    assert C_ == 3

    degrees = torch.tensor(
        [0] + [1] * 3 + [2] * 5 + [3] * 7, dtype=SH.dtype, device=SH.device
    )
    reg_weights = torch.exp(degrees)

    W_diag = lam * reg_weights.unsqueeze(0).expand(B, -1)
    Y = SH
    Y_T = Y.transpose(1, 2)
    lhs = torch.bmm(Y_T, Y)

    eye = torch.eye(D, dtype=SH.dtype, device=SH.device).unsqueeze(0).expand(B, -1, -1)
    lhs += eye * W_diag.unsqueeze(2)

    rhs = torch.bmm(Y_T, C)  # [B, D, 3]

    coeffs = torch.linalg.solve(lhs, rhs)  # [B, D, 3]

    return coeffs


def get_SF_coeffs(angles, colors, weights, basis_dim=16):
    colors = rgb_to_sh(colors)
    dirs = torch.stack(
        (
            torch.sin(angles[:, :, 0]) * torch.cos(angles[:, :, 1]),
            torch.sin(angles[:, :, 0]) * torch.sin(angles[:, :, 1]),
            torch.cos(angles[:, :, 0]),
        ),
        axis=-1,
    )
    sh_bases = _eval_sh_bases_fast(basis_dim, dirs)
    weights = weights[..., None]
    sh_bases *= weights
    colors *= weights
    coeffs = batched_sh_least_squares(sh_bases, colors)
    coeffs *= 0.28209479177387814  # make it supported by the rasterization code
    return coeffs


def transform_shs(shs_feat, rotation_matrix):
    P = torch.tensor(
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        dtype=rotation_matrix.dtype,
        device=rotation_matrix.device,
    )
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix.cpu())

    D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2]).to(
        device=rotation_matrix.device
    )
    D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2]).to(
        device=rotation_matrix.device
    )
    D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2]).to(
        device=rotation_matrix.device
    )

    shs_feat_1 = D_1 @ shs_feat[:, 1:4]
    shs_feat_2 = D_2 @ shs_feat[:, 4:9]
    shs_feat_3 = D_3 @ shs_feat[:, 9:]
    shs_feat = torch.concatenate(
        [shs_feat[:, :1], shs_feat_1, shs_feat_2, shs_feat_3], dim=1
    )
    return shs_feat


if __name__ == "__main__":
    pass
