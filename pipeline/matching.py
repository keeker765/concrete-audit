"""匹配过滤与评分"""
import cv2
import numpy as np
import torch
from .config import MIN_MATCHES, SCORE_W_CONF, SCORE_W_IR


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
