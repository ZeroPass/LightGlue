import torch
import cv2
import numpy as np
from lightglue import SuperPoint
from lightglue.utils import rbd


class SuperPointWithED(SuperPoint):
    def __init__(self, config, ed_config):
        super().__init__(**config)
        self.ed_conf = ed_config

    def run_edgedrawing(self, image_tensor):
        """Run EDLines on a torch image (1, 1, H, W) and return endpoints (N, 2)."""
        # Convert to uint8 for OpenCV
        img_np = (image_tensor[0, 0].cpu().numpy() * 255).astype(np.uint8)

        # Configure Edge Drawing (LSD Gradients)
        ed = cv2.ximgproc.createEdgeDrawing()
        params = cv2.ximgproc.EdgeDrawing_Params()
        params.EdgeDetectionOperator = cv2.ximgproc.EdgeDrawing.LSD
        params.GradientThresholdValue = self.ed_conf.get('gradient_threshold', 20)
        params.AnchorThresholdValue = self.ed_conf.get('anchor_threshold', 0)
        params.MinPathLength = self.ed_conf.get('min_length', 15)
        params.Sigma = self.ed_conf.get('sigma', 1.0)
        ed.setParams(params)

        # Detect
        ed.detectEdges(img_np)
        lines = ed.detectLines()

        if lines is None:
            return torch.zeros((0, 2), device=image_tensor.device)

        # Extract Endpoints [x1, y1, x2, y2] -> Stack into (N, 2)
        p1 = lines[:, :2]
        p2 = lines[:, 2:4]
        points = np.vstack((p1, p2))

        # Remove duplicates
        points = np.unique(points, axis=0)
        return torch.from_numpy(points).float().to(image_tensor.device)

    def forward(self, data):
        # 1. Run SuperPoint Backbone (get dense descriptor map)
        # We assume input is standard dictionary {'image': ...}
        if self.conf.fix_sampling:
            data = {k: v.clone() for k, v in data.items()}  # Safety copy

        # Run internal SuperPoint logic to get the raw maps
        # Note: We rely on the parent class methods to handle the backbone
        # But LightGlue's SuperPoint forward() is monolithic.
        # We will wrap the `forward` logic manually to intervene.

        image = data['image']
        if image.shape[1] == 3:
            # Convert RGB to Grayscale
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        # Get raw outputs from the underlying network
        # The 'net' is the actual CNN in LightGlue's SuperPoint wrapper
        pred = self.net(image)  # {'semi', 'desc'}

        # --- KEYPOINT SELECTION ---
        mode = self.ed_conf.get('mode', 'superpoint')

        # A. Get Standard SuperPoint Keypoints
        sp_keypoints = torch.zeros((0, 2), device=image.device)
        if mode in ['superpoint', 'combined']:
            # Standard NMS and extraction logic
            prob = pred['semi']
            if self.conf.nms_radius > 0:
                prob = self.simple_nms(prob, self.conf.nms_radius)

            # Extract top K
            keypoints = torch.nonzero(prob > self.conf.detection_threshold)
            scores = prob[keypoints[:, 0], keypoints[:, 1], keypoints[:, 2]]

            # Keep top-k
            if (
                self.conf.max_num_keypoints > 0
                and len(keypoints) > self.conf.max_num_keypoints
            ):
                scores, indices = torch.topk(scores, self.conf.max_num_keypoints)
                keypoints = keypoints[indices]

            # Convert (b, y, x) to (b, x, y)
            sp_keypoints = torch.flip(keypoints[:, 1:], [1]).float()

        # B. Get EdgeDrawing Keypoints
        ed_keypoints = torch.zeros((0, 2), device=image.device)
        if mode in ['edgedrawing', 'combined']:
            # We only handle batch size 1 for ED logic simplicity
            ed_pts = self.run_edgedrawing(image)
            ed_keypoints = ed_pts

        # C. Combine
        if mode == 'superpoint':
            final_kpts = sp_keypoints
        elif mode == 'edgedrawing':
            final_kpts = ed_keypoints
        else:  # combined
            if len(sp_keypoints) > 0 and len(ed_keypoints) > 0:
                final_kpts = torch.cat([sp_keypoints, ed_keypoints], dim=0)
            elif len(ed_keypoints) > 0:
                final_kpts = ed_keypoints
            else:
                final_kpts = sp_keypoints

        # Add batch dimension for sampling
        # (N, 2) -> (1, N, 2)
        if len(final_kpts) == 0:
            # Fallback to avoid crashes if image is empty
            final_kpts = torch.zeros((1, 0, 2), device=image.device)
        else:
            final_kpts = final_kpts.unsqueeze(0)

        # --- DESCRIPTOR SAMPLING ---
        # Crucial Step: We use SuperPoint's map to describe OUR custom points
        # sample_descriptors expects (B, C, H, W) and (B, N, 2)
        desc = self.sample_descriptors(pred['desc'], final_kpts)

        # Dummy scores for ED points (set to 1.0) if needed, or sample from probability map
        # Ideally, we sample scores from the probability map 'semi'
        scores = self.sample_descriptors(pred['semi'], final_kpts).squeeze(1)  # (1, N)

        return {
            'keypoints': final_kpts,  # (1, N, 2)
            'descriptors': desc,  # (1, N, 256)
            'keypoint_scores': scores,  # (1, N)
            'image_size': torch.tensor(image.shape[-2:][::-1], device=image.device)
            .unsqueeze(0),
        }
