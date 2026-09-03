# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning-free helpers for running Fast3R inference.

Everything needed to go from a raw ``Fast3R`` network + ``inference()`` output
to camera poses / aligned point clouds lives here, so that importing the
inference path never pulls in ``lightning`` / ``pl_bolts``. The training-time
:class:`~fast3r.models.multiview_dust3r_module.MultiViewDUSt3RLitModule` re-uses
these functions.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List

import numpy as np
import roma
import torch

from fast3r.dust3r.cloud_opt.init_im_poses import fast_pnp
from fast3r.dust3r.post_process import estimate_focal_knowing_depth_and_confidence_mask
from fast3r.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def align_local_pts3d_to_global(preds, views, min_conf_thr_percentile=0):
    """
    Aligns the local point clouds to the global coordinate frame.

    Args:
        preds (List[Dict]): A list of dictionaries containing predictions for each view.
        views (List[Dict]): A list of dictionaries containing ground truth data for each view.
        min_conf_thr_percentile (float): Minimum confidence percentile threshold (default is 0).

    Modifies:
        preds: Each pred dictionary in the list will have a new key 'pts3d_local_aligned_to_global',
            which contains the aligned local points.
    """
    # Check if required keys are present in preds
    for pred in preds:
        if 'pts3d_local' not in pred:
            raise ValueError("Key 'pts3d_local' not found in preds.")
        if 'conf_local' not in pred:
            raise ValueError("Key 'conf_local' not found in preds.")
        if 'pts3d_in_other_view' not in pred:
            raise ValueError("Key 'pts3d_in_other_view' not found in preds.")
        if 'conf' not in pred:
            raise ValueError("Key 'conf' (global head confidence) not found in preds.")

    num_views = len(preds)
    B, H, W, _ = preds[0]['pts3d_local'].shape  # Get batch size and dimensions

    # Function to process a single (view_index, batch_index) pair
    def process_view_batch(view_index, batch_index):
        pred = preds[view_index]
        view = views[view_index]

        # Get the predicted points from local and global heads for this sample
        pts3d_local = pred['pts3d_local'][batch_index]            # Shape: (H, W, 3)
        conf_local = pred['conf_local'][batch_index]              # Shape: (H, W)
        pts3d_global = pred['pts3d_in_other_view'][batch_index]   # Shape: (H, W, 3)
        conf_global = pred['conf'][batch_index]                   # Shape: (H, W)

        H_cur, W_cur, _ = pts3d_local.shape

        # Get valid_mask if it exists
        if 'valid_mask' in view:
            valid_mask = view['valid_mask'][batch_index]          # Shape: (H, W)
        else:
            valid_mask = torch.ones_like(conf_global, dtype=torch.bool)

        # Flatten the confidences to compute the threshold
        conf_global_flat = conf_global.reshape(-1)  # Shape: (N,)

        # Compute the confidence threshold
        conf_threshold_value = torch.quantile(conf_global_flat, min_conf_thr_percentile / 100.0)

        # Create a mask for high-confidence points
        conf_mask = conf_global >= conf_threshold_value

        # Combine masks
        final_mask = conf_mask & valid_mask  # Shape: (H, W)

        # Flatten the points and masks
        pts_local_flat = pts3d_local.view(-1, 3)   # Shape: (N, 3)
        pts_global_flat = pts3d_global.view(-1, 3) # Shape: (N, 3)
        final_mask_flat = final_mask.view(-1)      # Shape: (N,)

        # Select valid points
        x = pts_local_flat[final_mask_flat]    # Local points (M, 3)
        y = pts_global_flat[final_mask_flat]   # Global points (M, 3)
        # w = conf_global.view(-1)[final_mask_flat]  # Weights (M,)

        # Check if we have enough points after applying confidence threshold
        if x.shape[0] < 3:
            # Not enough points after applying confidence threshold
            # Use only valid_mask
            final_mask = valid_mask
            final_mask_flat = final_mask.view(-1)

            # Re-select points without confidence threshold
            x = pts_local_flat[final_mask_flat]    # Local points (M, 3)
            y = pts_global_flat[final_mask_flat]   # Global points (M, 3)
            # w = conf_global.view(-1)[final_mask_flat]  # Weights (M,)

        # Check again if we have enough points
        if x.shape[0] < 3:
            # Not enough points even after using valid_mask only
            # Use identity transformation
            R = torch.eye(3, device=pts_local_flat.device, dtype=pts_local_flat.dtype)
            t = torch.zeros(3, device=pts_local_flat.device, dtype=pts_local_flat.dtype)
            s = 1.0
        else:
            # Compute the rigid transformation with scaling
            R, t, s = roma.rigid_points_registration(
                x, y, compute_scaling=True
            )

        # Apply the transformation to all local points (including invalid ones)
        pts_local_aligned_flat = s * (pts_local_flat @ R.T) + t  # Shape: (N, 3)

        # Reshape back to (H, W, 3)
        pts_local_aligned = pts_local_aligned_flat.view(H_cur, W_cur, 3)

        return (view_index, batch_index, pts_local_aligned)

    # Create a list of all tasks (view_index, batch_index) pairs
    tasks = [(view_idx, batch_idx) for view_idx in range(num_views) for batch_idx in range(B)]

    # Use ThreadPoolExecutor to parallelize across tasks
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_view_batch, view_idx, batch_idx) for view_idx, batch_idx in tasks]

        # Collect the results
        results = [future.result() for future in futures]

    # Organize the results and update preds
    # Create a dictionary to store aligned points for each view
    aligned_pts_dict = {view_idx: [None] * B for view_idx in range(num_views)}

    for view_index, batch_index, pts_local_aligned in results:
        aligned_pts_dict[view_index][batch_index] = pts_local_aligned

    # Update preds with the aligned points
    for view_index in range(num_views):
        pred = preds[view_index]
        # Stack the aligned points back into a tensor of shape (B, H, W, 3)
        pred['pts3d_local_aligned_to_global'] = torch.stack(aligned_pts_dict[view_index], dim=0)


def estimate_camera_poses(preds, views=None, niter_PnP=10, focal_length_estimation_method='individual'):
    """Estimate camera poses and focal lengths using fast_pnp in parallel."""

    batch_size = len(preds[0]["pts3d_in_other_view"])  # Get the batch size

    # Prepare data_for_processing
    data_for_processing = []

    for i in range(batch_size):
        # Collect preds for each sample in the batch
        sample_preds = [{key: value[i].cpu() for key, value in view.items()} for view in preds]

        data_for_processing.append(sample_preds)

    # Estimate the focal length
    def estimate_focal_for_sample(sample_preds):
        if focal_length_estimation_method == 'first_view_from_global_head':
            # Use global head outputs for focal length estimation
            pts3d_i = sample_preds[0]["pts3d_in_other_view"].unsqueeze(0)  # Shape: (1, H, W, 3)
            conf_i = sample_preds[0]["conf"].unsqueeze(0)                  # Shape: (1, H, W)
        elif focal_length_estimation_method == 'first_view_from_local_head':
            # Use local head outputs for focal length estimation
            pts3d_i = sample_preds[0]["pts3d_local_aligned_to_global"].unsqueeze(0)  # Shape: (1, H, W, 3)
            conf_i = sample_preds[0]["conf_local"].unsqueeze(0)                       # Shape: (1, H, W)
        elif focal_length_estimation_method == 'individual':
            # Focal length will be estimated individually per view
            return sample_preds
        else:
            raise ValueError(f"Unknown focal_length_estimation_method: {focal_length_estimation_method}")

        # Estimate focal length using the provided function and confidence mask
        estimated_focal = estimate_focal(pts3d_i, conf_i, min_conf_thr_percentile=10)

        # Store the estimated focal length in sample_preds
        for view_pred in sample_preds:
            view_pred["focal_length"] = estimated_focal
            # view_pred["focal_length"] = 256.64

        return sample_preds

    with ThreadPoolExecutor() as executor:
        data_for_processing = list(executor.map(estimate_focal_for_sample, data_for_processing))

    # Estimate the camera poses
    # Use ProcessPoolExecutor to parallelize processing across samples in the batch
    poses_c2w_all = []
    estimated_focals_all = []

    # Use partial to fix arguments
    estimate_cam_pose_one_sample_partial = partial(estimate_cam_pose_one_sample, niter_PnP=niter_PnP, min_conf_thr_percentile=85)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(estimate_cam_pose_one_sample_partial, data_for_processing))

    # Collect results from all processed samples
    for poses_c2w_sample, estimated_focals_sample in results:
        poses_c2w_all.append(poses_c2w_sample)
        estimated_focals_all.append(estimated_focals_sample)

    return poses_c2w_all, estimated_focals_all


def correct_preds_orientation(preds, views):
    # *In-place* correction of the orientation of the predicted points and confidence maps

    # correct the shape of the predicted points and confidence maps if the view is portrait
    # this is because the data loader transposed the input images and valid_masks to landscape
    # see datasets/base/base_stereo_view_dataset.py
    if views is not None:
        for pred, view in zip(preds, views):
            # debug: use GT point map to estimate poses
            # pred["pts3d_in_other_view"] = view["pts3d"]  # shape (B, H, W, 3)
            # pred["conf"] = view['valid_mask'].float() if "valid_mask" in view else torch.ones_like(pred["conf"])  # shape (B, H, W)
            # pred["focal_length"] = view["camera_intrinsics"][:, 0, :2].sum(1)
            # end debug

            # check if the view is protrait or landscape (true_shape: (H, W))
            conf_list = []
            pts3d_list = []

            for i in range(view["true_shape"].shape[0]):
                H, W = view["true_shape"][i]
                if H > W:  # portrait
                    # Transpose the tensors
                    transposed_conf = pred["conf"][i].transpose(0, 1)
                    transposed_pts3d = pred["pts3d_in_other_view"][i].transpose(0, 1)

                    # Append the transposed tensors to the lists
                    conf_list.append(transposed_conf)
                    pts3d_list.append(transposed_pts3d)
                else:
                    # Append the original tensors to the lists
                    conf_list.append(pred["conf"][i])
                    pts3d_list.append(pred["pts3d_in_other_view"][i])

            pred["conf"] = conf_list
            pred["pts3d_in_other_view"] = pts3d_list

            if "pts3d_local" in pred:
                conf_local_list = []
                pts3d_local_list = []
                if "pts3d_local_aligned_to_global" in pred:
                    pts3d_local_aligned_to_global_list = []

                for i in range(view["true_shape"].shape[0]):
                    H, W = view["true_shape"][i]
                    if H > W:
                        # Transpose the tensors
                        transposed_conf_local = pred["conf_local"][i].transpose(0, 1)
                        transposed_pts3d_local = pred["pts3d_local"][i].transpose(0, 1)
                        if "pts3d_local_aligned_to_global" in pred:
                            transposed_pts3d_local_aligned_to_global = pred["pts3d_local_aligned_to_global"][i].transpose(0, 1)

                        # Append the transposed tensors to the lists
                        conf_local_list.append(transposed_conf_local)
                        pts3d_local_list.append(transposed_pts3d_local)
                        if "pts3d_local_aligned_to_global" in pred:
                            pts3d_local_aligned_to_global_list.append(transposed_pts3d_local_aligned_to_global)
                    else:
                        # Append the original tensors to the lists
                        conf_local_list.append(pred["conf_local"][i])
                        pts3d_local_list.append(pred["pts3d_local"][i])
                        if "pts3d_local_aligned_to_global" in pred:
                            pts3d_local_aligned_to_global_list.append(pred["pts3d_local_aligned_to_global"][i])

                pred["conf_local"] = conf_local_list
                pred["pts3d_local"] = pts3d_local_list
                if "pts3d_local_aligned_to_global" in pred:
                    pred["pts3d_local_aligned_to_global"] = pts3d_local_aligned_to_global_list


def estimate_cam_pose_one_sample(sample_preds, device='cpu', niter_PnP=10, min_conf_thr_percentile=0):
    poses_c2w = []
    estimated_focals = []

    # Define the function to process each view
    def process_view(view_idx):
        pts3d = sample_preds[view_idx]["pts3d_in_other_view"].cpu().numpy().squeeze()  # (H, W, 3)
        valid_mask = sample_preds[view_idx]["conf"].cpu().numpy().squeeze() > 1.0  # Confidence mask
        # use the confidence map to filter out low-confidence points
        # conf_threshold_value = torch.quantile(sample_preds[view_idx]["conf"].view(-1), min_conf_thr_percentile / 100.0)
        # valid_mask = sample_preds[view_idx]["conf"].cpu().numpy().squeeze() >= float(conf_threshold_value ) # Confidence mask
        focal_length = float(sample_preds[view_idx]["focal_length"]) if "focal_length" in sample_preds[view_idx] else None

        # Call fast_pnp with unflattened pts3d and mask
        focal_length, pose_c2w = fast_pnp(
            torch.tensor(pts3d),
            focal_length,  # Guess focal length
            torch.tensor(valid_mask, dtype=torch.bool),
            "cpu",
            pp=None,  # Use default principal point (center of image)
            niter_PnP=niter_PnP
        )

        if pose_c2w is None or focal_length is None:
            log.warning(f"Failed to estimate pose for view {view_idx}")
            return np.eye(4), focal_length  # Return identity pose in case of failure

        # Return the results for this view
        return pose_c2w.cpu().numpy(), focal_length

    # Use ThreadPoolExecutor to process views in parallel
    with ThreadPoolExecutor() as executor:
        # Map the process_view function to each view index
        results = list(executor.map(process_view, range(len(sample_preds))))

    # Collect the results
    for pose_c2w_result, focal_length_result in results:
        poses_c2w.append(pose_c2w_result)
        estimated_focals.append(focal_length_result)

    return poses_c2w, estimated_focals


def estimate_focal(pts3d_i, conf_i, pp=None, min_conf_thr_percentile=10):
    B, H, W, THREE = pts3d_i.shape
    assert B == 1  # Since we're processing one sample at a time

    if pp is None:
        pp = torch.tensor((W / 2, H / 2), device=pts3d_i.device).view(1, 2)  # Shape: (1, 2)

    # Flatten the confidence map using reshape instead of view
    conf_flat = conf_i.reshape(-1)

    # Compute the confidence threshold based on the percentile
    percentile = min_conf_thr_percentile / 100.0  # Convert to a fraction
    conf_threshold = torch.quantile(conf_flat, percentile)

    # Create the confidence mask based on the computed threshold
    conf_mask = conf_i >= conf_threshold
    conf_mask = conf_mask.view(B, H, W)  # Ensure shape is (B, H, W)

    # Check if there are enough valid points
    if conf_mask.sum() < 10:  # Adjust the minimum number as needed
        print("Not enough high-confidence points for focal estimation.")
        # Optionally, adjust the percentile or set conf_mask to all True
        # For example:
        # conf_mask = torch.ones_like(conf_mask, dtype=torch.bool)

    focal = estimate_focal_knowing_depth_and_confidence_mask(
        pts3d_i, pp.unsqueeze(0), conf_mask, focal_mode="weiszfeld"
    ).ravel()
    return float(focal)


class MultiViewDUSt3RInferenceWrapper:
    """Lightning-free stand-in for ``MultiViewDUSt3RLitModule`` for inference.

    Wraps a raw :class:`~fast3r.models.fast3r.Fast3R` network and exposes the
    subset of the LightningModule API that the demo / notebooks use:
    ``load_for_inference``, ``eval``, ``forward``/``__call__``,
    ``align_local_pts3d_to_global``, ``estimate_camera_poses`` and
    ``correct_preds_orientation``.
    """

    def __init__(self, net):
        self.net = net

    @classmethod
    def load_for_inference(cls, net):
        net.eval()
        return cls(net)

    def eval(self):
        self.net.eval()
        return self

    def train(self, mode: bool = True):
        self.net.train(mode)
        return self

    def to(self, *args, **kwargs):
        self.net.to(*args, **kwargs)
        return self

    @property
    def device(self):
        return next(self.net.parameters()).device

    def forward(self, views: List[dict]):
        return self.net(views)

    __call__ = forward

    # Expose the module-level helpers as (static) methods for API compatibility.
    align_local_pts3d_to_global = staticmethod(align_local_pts3d_to_global)
    estimate_camera_poses = staticmethod(estimate_camera_poses)
    correct_preds_orientation = staticmethod(correct_preds_orientation)
