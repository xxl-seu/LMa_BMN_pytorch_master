# _*_ coding:utf-8 _*_
import numpy as np
from scipy.integrate import simps

def compute_AR(video_lst, daps_results, ground_truth, set_proposals_mae, set_proposals_me, method):
    # For each video, computes tiou scores among the retrieved proposals.
    score_lst_mae = []
    score_lst_me = []
    total_gt_num_mae = 0.
    total_gt_num_me = 0.
    total_pr_num_mae = 0.
    total_pr_num_me = 0.
    for videoid in video_lst:
        this_video_proposals_me = [] # proposals_me
        this_video_proposals_mae = [] # proposals_mae
        for i in range(len(daps_results[videoid])):
            if daps_results[videoid][i]['segment'][1] - daps_results[videoid][i]['segment'][0] > 0.5:
                this_video_proposals_mae.append(daps_results[videoid][i]['segment'])
            else:
                this_video_proposals_me.append(daps_results[videoid][i]['segment'])
        this_video_proposals_me = np.array(this_video_proposals_me)
        this_video_proposals_mae = np.array(this_video_proposals_mae)

        this_video_ground_truth_me = [] # GTs_me
        this_video_ground_truth_mae = [] # GTs_mae
        for i in range(len(ground_truth[videoid]['annotations'])):
            if ground_truth[videoid]['annotations'][i]['segment'][1] - ground_truth[videoid]['annotations'][i]['segment'][0] > 0.5:
                this_video_ground_truth_mae.append(ground_truth[videoid]['annotations'][i]['segment'])
            else:
                this_video_ground_truth_me.append(ground_truth[videoid]['annotations'][i]['segment'])
        this_video_ground_truth_mae = np.array(this_video_ground_truth_mae)
        this_video_ground_truth_me = np.array(this_video_ground_truth_me)
        tiou_mae, gt_num_mae, pr_num_mae = segment_tiou(this_video_ground_truth_mae, this_video_proposals_mae)
        tiou_me, gt_num_me, pr_num_me = segment_tiou(this_video_ground_truth_me, this_video_proposals_me)
        #print(tiou.shape, gt_num, pr_num)
        score_lst_mae.append(tiou_mae) # 包含每个video mae结果的列表
        score_lst_me.append(tiou_me) # 包含每个video me结果的列表
        total_gt_num_mae += gt_num_mae # 总gt mae数
        total_gt_num_me += gt_num_me # 总gt me数
        total_pr_num_mae += pr_num_mae # 总proposal mae数
        total_pr_num_me += pr_num_me # 总proposal me数
    if method == 1: # 单个计算
        matches_mae = np.empty(len(video_lst))
        matches_me = np.empty(len(video_lst))
        for i, score in enumerate(score_lst_mae):
            matches_mae[i] = ((score[:, :set_proposals_mae] >= 0.5).sum(axis=1) > 0).sum()  # 表示每个video中的gt mae被找出的个数
        for i, score in enumerate(score_lst_me):
            matches_me[i] = ((score[:, :min(set_proposals_me, score.shape[1])] >= 0.5).sum(axis=1) > 0).sum()  # 表示每个video中的gt被找出的个数
    else: # 集体验证
        matches_mae = np.empty(len(video_lst))
        matches_me = np.empty(len(video_lst))
        for i, score in enumerate(score_lst_mae):
            matches_mae[i] = ((score[:, :set_proposals_mae[i]] >= 0.5).sum(axis=1) > 0).sum()  # 表示每个video中的gt被找出的个数
        for i, score in enumerate(score_lst_me):
            matches_me[i] = ((score[:, :min(set_proposals_me[i], score.shape[1])] >= 0.5).sum(axis=1) > 0).sum()  # 表示每个video中的gt被找出的个数

    # 计算Recall
    Recall_mae = matches_mae.sum() / (total_gt_num_mae + 1e-10)
    Recall_me = matches_me.sum() / (total_gt_num_me + 1e-10)
    Recall = (matches_mae.sum() + matches_me.sum()) / (total_gt_num_mae + total_gt_num_me + 1e-10)
    # 计算Precision
    if method == 1:
        Precision_mae = matches_mae.sum() / (set_proposals_mae*len(video_lst) + 1e-10)
        Precision_me = matches_me.sum() / (set_proposals_me*len(video_lst) + 1e-10)
        Precision = (matches_mae.sum() + matches_me.sum()) / ((set_proposals_mae + set_proposals_me)*len(video_lst) + 1e-10)
    else:
        Precision_mae = matches_mae.sum() / (sum(set_proposals_mae) + 1e-10)
        Precision_me = matches_me.sum() / (sum(set_proposals_me) + 1e-10)
        Precision = (matches_mae.sum() + matches_me.sum()) / (sum(set_proposals_mae) + sum(set_proposals_me) + 1e-10)
    # 计算F1-score
    F1_score_mae = 2 * Recall_mae * Precision_mae / (Recall_mae + Precision_mae + 1e-10)
    F1_score_me = 2 * Recall_me * Precision_me / (Recall_me + Precision_me + 1e-10)
    F1_score = 2*Recall*Precision/(Recall+Precision+1e-10)
    return Recall, Recall_mae, Recall_me, Precision, Precision_mae, Precision_me, F1_score, F1_score_mae, F1_score_me



def segment_tiou(target_segments, test_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    test_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [m x n] with IOU ratio.
    Note: It assumes that target-segments are more scarce that test-segments
    """
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        m, n = target_segments.shape[0], test_segments.shape[0]
        tiou = np.zeros((1, 1))
        return tiou, m, n
        # raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    tiou = np.empty((m, n))
    for i in range(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0]) +
                 (target_segments[i, 1] - target_segments[i, 0]) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        tiou[i, :] = intersection / union
    return tiou, m, n
