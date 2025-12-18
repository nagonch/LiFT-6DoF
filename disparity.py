from plenpy.lightfields import LightField
import numpy as np
import cv2
import torch
import warnings

warnings.filterwarnings("ignore")


def robust_sigma_mad(x):
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def get_disparity_range(disparity, sigma=1.5):
    disparity = disparity.reshape(-1)
    disp_centroid = np.median(disparity)
    disp_std = robust_sigma_mad(disparity)
    disp_min, disp_max = (
        disp_centroid - sigma * disp_std,
        disp_centroid + sigma * disp_std,
    )
    return disp_min, disp_max


def weighted_fusion(disp, conf):
    confidence = np.nanmean(conf**2, axis=-1)
    disparity = np.nanmean(disp * conf**2, axis=-1) / confidence
    confidence = np.sqrt(confidence)
    return disparity, confidence


def denoise_disparity(disparity, disp_min, disp_max, denoise_param=35):
    disparity = np.nan_to_num(disparity, nan=0.0)
    disparity = (disparity - disp_min) / (disp_max - disp_min) * 255
    disparity = disparity.astype(np.uint8)
    disparity = cv2.fastNlMeansDenoising(disparity, h=denoise_param)
    disparity = disparity / 255 * (disp_max - disp_min) + disp_min
    return disparity


def get_LF_disparity(LF, denoise_param=35, sigma=1.5):
    disparity, confidence = LightField(LF).get_disparity(
        vmin=-100, vmax=100, fusion_method="no_fusion"
    )
    disparity = np.abs(disparity)
    disparity = np.nan_to_num(disparity, nan=0.0)
    disp_min, disp_max = get_disparity_range(disparity, sigma=sigma)
    disparity = np.clip(disparity, disp_min, disp_max)
    disparity, confidence = weighted_fusion(disparity, confidence)
    disparity = denoise_disparity(
        disparity,
        disp_min,
        disp_max,
        denoise_param=denoise_param,
    )
    return disparity, confidence


def fuse_disparities(
    disparity,
    dam_disparity,
    sanity_mask,
    max_disparity=100,
):
    sanity_mask = sanity_mask & (disparity < max_disparity)
    reliable_disparities = disparity[sanity_mask].reshape(-1).float()
    corresponding_dam_disparities = dam_disparity[sanity_mask].reshape(-1).float()
    X = torch.stack(
        [corresponding_dam_disparities, torch.ones_like(corresponding_dam_disparities)],
        dim=1,
    )
    sol = torch.linalg.lstsq(X, reliable_disparities).solution
    alpha, beta = sol[0], sol[1]
    result_disparities = alpha * dam_disparity + beta
    return result_disparities


def get_frame_disparity(frame, min_fit_confidence=0.9):
    LF = frame["LF"]
    dam_disparity = frame["dam_disparity"]

    LF_disparity, confidence = get_LF_disparity(
        LF.cpu().numpy(),
    )
    LF_disparity, confidence = torch.tensor(
        LF_disparity, device=dam_disparity.device
    ), torch.tensor(confidence, device=dam_disparity.device)
    sanity_mask = confidence > min_fit_confidence
    result_disparity = fuse_disparities(
        LF_disparity, dam_disparity, sanity_mask=sanity_mask
    )
    return result_disparity


if __name__ == "__main__":
    pass
