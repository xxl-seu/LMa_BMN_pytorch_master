# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import multiprocessing as mp

from utils import iou_with_anchors


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def getDatasetDict(opt):
    df = pd.read_csv(opt["video_info"]+ '.csv')
    json_data = load_json(opt["video_anno"])
    database = json_data
    video_dict = {}
    for i in range(len(df)):
        video_name = df.video.values[i]
        video_info = database[video_name]
        video_new_info = {}
        video_new_info['duration_frame'] = video_info['duration_frame']
        video_new_info['duration_second'] = video_info['duration_second']
        video_new_info["feature_frame"] = video_info['feature_frame']
        video_subset = df.subset.values[i]
        video_new_info['annotations'] = video_info['annotations']
        if video_subset == 'testing':
            video_dict[video_name] = video_new_info
    return video_dict

def soft_nms(df, alpha, threhold):
    '''
    df: proposals generated by network;
    alpha: alpha value of Gaussian decaying function;
    t1, t2: threshold for soft nms.
    '''
    df = df.sort_values(by="score", ascending=False) # 将df按照score进行了排序
    tstart = list(df.xmin.values[:])
    # print('tstart:', tstart)
    tend = list(df.xmax.values[:])
    # print('tend:', len(tend))
    tscore = list(df.score.values[:])
    # print('tscore:', len(tscore))

    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 1 and len(rscore) < 301:
        max_index = tscore.index(max(tscore)) # 返回最大置信度区间的索引
        tmp_iou_list = iou_with_anchors(
            np.array(tstart),
            np.array(tend), tstart[max_index], tend[max_index]) # tstart开始时间，tend结束时间，置信度最大区间的开始时间，置信度最大区间的结束时间
        # print(tmp_iou_list) # 分别计算最大置信度的区间和其他每个区间的iou，存储在tmp_iou_list中
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = tmp_iou_list[idx]
                if tmp_iou > threhold:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / alpha)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def video_post_process(opt, video_list, video_dict, threhold):
    for video_name in video_list:
        df = pd.read_csv("./output/Lma_BMN_results/"+opt["dataset"]+'/' + video_name + ".csv")

        if len(df) > 1:
            snms_alpha = opt["soft_nms_alpha"]
            df = soft_nms(df, snms_alpha, threhold)

        df = df.sort_values(by="score", ascending=False)
        video_info = video_dict[video_name]
        if opt['dataset'] == 'CASME':
            interval = 4
        elif opt['dataset'] == 'SAMM':
            interval = 32
        else:
            print('Dataset invalid')
        video_duration = float(video_info["duration_frame"] // interval * interval) / video_info["duration_frame"] * video_info[
            "duration_second"] # ‘duration_frame’是真实的视频帧数，最后计算得到处理后的视频时长
        proposal_list = []

        for j in range(min(300, len(df))):
            tmp_proposal = {}
            proposal_frames_min = opt['proposal_frames_min']
            proposal_frames_max = opt['proposal_frames_max']
            if proposal_frames_min <= (min(1, df.xmax.values[j]) - max(0, df.xmin.values[j])) * video_duration <= proposal_frames_max:
                tmp_proposal["score"] = df.score.values[j]
                tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                           min(1, df.xmax.values[j]) * video_duration]
                proposal_list.append(tmp_proposal)
        result_dict[video_name[:]] = proposal_list


def Lma_BMN_post_processing(opt): # 将高置信度的proposal按置信度从高到低输出在json文件中
    video_dict = getDatasetDict(opt)
    video_list = list(video_dict.keys())  # [:100]
    # print(video_list)
    global result_dict
    result_dict = mp.Manager().dict()

    num_videos = len(video_list)
    num_videos_per_thread = num_videos // opt["post_process_thread"]
    processes = []
    for tid in range(opt["post_process_thread"] - 1):
        tmp_video_list = video_list[tid * num_videos_per_thread:(tid + 1) * num_videos_per_thread]
        p = mp.Process(target=video_post_process, args=(opt, tmp_video_list, video_dict, opt['soft_nms_threshold']))
        p.start()
        processes.append(p)
    tmp_video_list = video_list[(opt["post_process_thread"] - 1) * num_videos_per_thread:]
    p = mp.Process(target=video_post_process, args=(opt, tmp_video_list, video_dict, opt['soft_nms_threshold']))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    output_dict = {"version": opt['dataset'], "results": result_dict, "external_data": {}}
    outfile = open(opt["result_file"] + opt['dataset'] + '/' + 'result_proposal.json', "w")
    json.dump(output_dict, outfile)
    outfile.close()

# opt = opts.parse_opt()
# opt = vars(opt)
# BSN_post_processing(opt)
