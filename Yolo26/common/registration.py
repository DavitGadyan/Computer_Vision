"""Camera-motion compensation: keep zones defined on a reference frame aligned with a drifting camera (drone, PTZ)."""
import cv2
import numpy as np


class FrameRegistrar:
    def __init__(self, reference, n_features=3000, every=3, scale=0.5):
        self.orb = cv2.ORB_create(n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.scale, self.every = scale, every
        self.kp_ref, self.des_ref = self._features(reference)
        self.H = np.eye(3, dtype=np.float32)
        self.n = 0
        self.inliers = 0

    def _features(self, frame):
        g = cv2.cvtColor(cv2.resize(frame, None, fx=self.scale, fy=self.scale), cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(g, None)

    def update(self, frame):
        """Return homography mapping reference-frame coordinates -> current-frame coordinates."""
        self.n += 1
        if self.n % self.every:
            return self.H
        kp, des = self._features(frame)
        if des is None or self.des_ref is None:
            return self.H
        matches = sorted(self.matcher.match(self.des_ref, des), key=lambda m: m.distance)[:800]
        if len(matches) < 30:
            return self.H
        src = np.float32([self.kp_ref[m.queryIdx].pt for m in matches]) / self.scale
        dst = np.float32([kp[m.trainIdx].pt for m in matches]) / self.scale
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is not None and mask.sum() > 25:
            self.H = (0.6 * H + 0.4 * self.H).astype(np.float32)  # light temporal smoothing
            self.inliers = int(mask.sum())
        return self.H

    @staticmethod
    def warp(quad, H):
        return cv2.perspectiveTransform(np.asarray(quad, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
