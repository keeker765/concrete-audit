"""匹配过滤与评分"""
import cv2
import numpy as np
import torch
from .config import MIN_MATCHES, SCORE_W_CONF, SCORE_W_IR


def filter_matches_by_center_geometry(
    kpts0,
    kpts1,
    matches,
    scores,
    center0,
    center1,
    max_dr: float = 60.0,
    min_cos: float = 0.90,
):
    """Filter (or down-weight) matches using a local coordinate system around a center.

    This implements the user's requested constraints:
    - Euclidean distance consistency to center (|r0-r1| <= max_dr)
    - Direction consistency via cosine similarity (cos(v0,v1) >= min_cos)

    Args:
      kpts0,kpts1: torch keypoints [N,2] in image coords
      matches: torch int [M,2] indices
      scores: torch float [M] or None
      center0,center1: (x,y) in same coordinate system as keypoints.
      max_dr: allowed radial difference (pixels)
      min_cos: minimum cosine similarity of direction vectors

    Returns:
      matches_f, scores_f, stats(dict)
    """
    if matches is None or len(matches) == 0:
        return matches, scores, {"geo_kept": 0, "geo_total": 0, "geo_keep_ratio": 0.0}
    if center0 is None or center1 is None:
        # no center -> no geometry filter
        return matches, scores, {"geo_kept": int(len(matches)), "geo_total": int(len(matches)), "geo_keep_ratio": 1.0, "geo_skipped": True}

    c0x, c0y = float(center0[0]), float(center0[1])
    c1x, c1y = float(center1[0]), float(center1[1])

    pts0 = kpts0[matches[:, 0]].detach().cpu().numpy()
    pts1 = kpts1[matches[:, 1]].detach().cpu().numpy()

    v0 = pts0 - np.array([[c0x, c0y]], dtype=np.float32)
    v1 = pts1 - np.array([[c1x, c1y]], dtype=np.float32)

    r0 = np.linalg.norm(v0, axis=1)
    r1 = np.linalg.norm(v1, axis=1)
    dr = np.abs(r0 - r1)

    denom = (np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1) + 1e-8)
    cos = (v0[:, 0] * v1[:, 0] + v0[:, 1] * v1[:, 1]) / denom

    keep = np.where((dr <= max_dr) & (cos >= min_cos))[0]
    if keep.size == 0:
        keep_t = torch.empty((0,), dtype=torch.long)
    else:
        keep_t = torch.from_numpy(keep.astype(np.int64))

    matches_f = matches[keep_t]
    scores_f = scores[keep_t] if scores is not None and len(scores) == len(matches) else scores

    stats = {
        "geo_total": int(len(matches)),
        "geo_kept": int(len(matches_f)),
        "geo_keep_ratio": float(len(matches_f) / max(len(matches), 1)),
        "geo_mean_cos": float(np.mean(cos[keep]) if keep.size else 0.0),
        "geo_mean_dr": float(np.mean(dr[keep]) if keep.size else 0.0),
        "geo_min_cos": float(np.min(cos[keep]) if keep.size else 0.0),
        "geo_max_dr": float(np.max(dr[keep]) if keep.size else 0.0),
    }

    return matches_f, scores_f, stats


def filter_matches(kpts0, kpts1, matches, scores, mask0, mask1):
    """过滤落在排除区域（贴纸）内的匹配点，同步 scores。"""
    if len(matches) == 0:
        return matches, scores

    keep = []
    for i, m in enumerate(matches):
        p0 = kpts0[m[0]].int().detach().cpu().numpy()
        p1 = kpts1[m[1]].int().detach().cpu().numpy()
        in0 = (0 <= p0[1] < mask0.shape[0] and
               0 <= p0[0] < mask0.shape[1] and
               mask0[p0[1], p0[0]] > 0)
        in1 = (0 <= p1[1] < mask1.shape[0] and
               0 <= p1[0] < mask1.shape[1] and
               mask1[p1[1], p1[0]] > 0)
        if not in0 and not in1:
            keep.append(i)

    if not keep:
        return matches, scores

    keep_t = torch.tensor(keep)
    matches_f = matches[keep_t]
    scores_f = scores[keep_t] if scores is not None else None
    return matches_f, scores_f


def filter_points_by_mask(kpts0, kpts1, conf, mask0, mask1):
    """过滤 LoFTR 等 dense matcher 返回的点对（直接是坐标，不是索引）。"""
    if len(kpts0) == 0:
        return kpts0, kpts1, conf

    keep = []
    for i in range(len(kpts0)):
        p0 = kpts0[i].int().cpu().numpy()
        p1 = kpts1[i].int().cpu().numpy()
        in0 = (0 <= p0[1] < mask0.shape[0] and
               0 <= p0[0] < mask0.shape[1] and
               mask0[p0[1], p0[0]] > 0)
        in1 = (0 <= p1[1] < mask1.shape[0] and
               0 <= p1[0] < mask1.shape[1] and
               mask1[p1[1], p1[0]] > 0)
        if not in0 and not in1:
            keep.append(i)

    if not keep:
        return kpts0[:0], kpts1[:0], conf[:0]

    idx = torch.tensor(keep)
    return kpts0[idx], kpts1[idx], conf[idx]


def compute_score(kpts0, kpts1, matches, match_scores):
    """
    综合评分 = W_CONF×匹配置信度 + W_IR×RANSAC内点率
    匹配数 < MIN_MATCHES 时施加线性惩罚
    """
    n = len(matches)
    if n < 4:
        return 0.0, 0.0, 0.0, np.array([]), n

    pts0 = kpts0[matches[:, 0]].detach().cpu().numpy()
    pts1 = kpts1[matches[:, 1]].detach().cpu().numpy()

    if match_scores is not None and len(match_scores) == len(matches):
        mean_conf = match_scores.detach().cpu().float().mean().item()
    else:
        mean_conf = 0.5

    try:
        H, inliers_mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        if inliers_mask is None:
            inlier_ratio = 0.0
            inliers_mask = np.zeros(n, dtype=bool)
        else:
            inliers_mask = inliers_mask.ravel().astype(bool)
            inlier_ratio = inliers_mask.sum() / n
    except cv2.error:
        inlier_ratio = 0.0
        inliers_mask = np.zeros(n, dtype=bool)

    raw_score = SCORE_W_CONF * mean_conf + SCORE_W_IR * inlier_ratio

    if n < MIN_MATCHES:
        final_score = raw_score * (n / MIN_MATCHES)
    else:
        final_score = raw_score

    return final_score, mean_conf, inlier_ratio, inliers_mask, n


def compute_score_from_points(pts0, pts1, conf, min_matches=None):
    """直接从点对坐标计算分数（用于 LoFTR 等 dense matcher）。"""
    if min_matches is None:
        min_matches = MIN_MATCHES
    n = len(pts0)
    if n < 4:
        return 0.0, 0.0, 0.0, np.array([]), n

    mean_conf = float(conf.mean())

    try:
        H, inliers_mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        if inliers_mask is None:
            inlier_ratio = 0.0
            inliers_mask = np.zeros(n, dtype=bool)
        else:
            inliers_mask = inliers_mask.ravel().astype(bool)
            inlier_ratio = inliers_mask.sum() / n
    except cv2.error:
        inlier_ratio = 0.0
        inliers_mask = np.zeros(n, dtype=bool)

    raw_score = SCORE_W_CONF * mean_conf + SCORE_W_IR * inlier_ratio

    if n < min_matches:
        final_score = raw_score * (n / min_matches)
    else:
        final_score = raw_score

    return final_score, mean_conf, inlier_ratio, inliers_mask, n
