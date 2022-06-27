# -*- coding: utf-8 -*-
import sys
from dataset import VideoDataSet
from loss_function import Lma_BMN_loss_func, get_mask
import os
import json
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import opts
from models import Lma_BMN
import pandas as pd
from prediction import *
import pdb
import numpy as np
import random
from post_processing import Lma_BMN_post_processing
from eval import evaluation_proposal

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
sys.dont_write_bytecode = True

def train_Lma_BMN(data_loader, model, optimizer, epoch, bm_mask):
    model.train()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()
        confidence_map, start, end = model(input_data)
        loss = Lma_BMN_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

    print(
        "Lma_BMN training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/Lma_BMN_" + str(epoch).zfill(2) + ".pth.tar")

def Lma_BMN_Train(opt):
    model = Lma_BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],
                           weight_decay=opt["weight_decay"])

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="training"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    bm_mask = get_mask(opt["temporal_scale"])
    for epoch in range(opt["train_epochs"]):
        scheduler.step()
        train_Lma_BMN(train_loader, model, optimizer, epoch, bm_mask)

def Lma_BMN_inference(opt): # 输出每个样本的每个ph(ij)的置信度、开始结束的概率以及总的概率
    model = Lma_BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/Lma_BMN_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="testing"),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for idx, input_data in test_loader:
            video_name = test_loader.dataset.video_list[idx[0]]
            input_data = input_data.cuda()
            confidence_map, start, end = model(input_data)
            # print(start.shape,end.shape,confidence_map.shape)
            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()
            # 遍历起始分界点与结束分界点的组合
            new_props = []
            for idx in range(tscale):
                for jdx in range(tscale):
                    start_index = idx
                    end_index = jdx + 1
                    if start_index < end_index and  end_index<tscale :
                        xmin = start_index / tscale
                        xmax = end_index / tscale
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        clr_score = clr_confidence[idx, jdx]
                        reg_score = reg_confidence[idx, jdx]
                        score = xmin_score * xmax_score * np.sqrt(clr_score * reg_score)
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
            new_props = np.stack(new_props)
            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv("./output/Lma_BMN_results/" + opt["dataset"] + '/' + video_name + ".csv", index=False)


def main(opt):
    ##################################### train #####################################################################
    opt["mode"] = "train"
    Lma_BMN_Train(opt)
    ##################################### test #####################################################################
    opt["mode"] = "inference"
    Lma_BMN_inference(opt) # 输出每个样本的置信度、开始结束的概率以及总的概率
    Lma_BMN_post_processing(opt) # 将符合要求的高置信度的proposal按置信度从高到低输出在json文件中
    print("Post processing finished")


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()
    main(opt)
