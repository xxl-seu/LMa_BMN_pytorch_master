import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint')
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.001)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=20)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16)
    parser.add_argument(
        '--step_size',
        type=int,
        default=5)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)

    # Overall Dataset settings
    parser.add_argument(
        '--dataset',
        type=str,
        default="CASME")

    parser.add_argument(
        '--video_info',
        type=str,
        default="") # 训练集、测试集信息存放csv
    parser.add_argument(
        '--video_anno',
        type=str,
        default="") # 训练集视频onset、offset信息存放json
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=100)
    parser.add_argument(
        '--set_proposal',
        type=int,
        default=8)
    parser.add_argument(
        '--feature_path',
        type=str,
        default="") # 特征存放目录
    parser.add_argument(
        '--proposal_frames_min',
        type=float,
        default=0)
    parser.add_argument(
        '--proposal_frames_max',
        type=float,
        default=2)
    parser.add_argument(
        '--num_sample',
        type=int,
        default=32)
    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3)
    parser.add_argument(
        '--prop_boundary_ratio',
        type=int,
        default=0.5)

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=512)

    # Post processing
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=8)
    parser.add_argument(
        '--soft_nms_alpha',
        type=float,
        default=0.4)
    parser.add_argument(
        '--soft_nms_threshold',
        type=float,
        default=0.1)
    parser.add_argument(
        '--result_file',
        type=str,
        default="./output/Lma_BMN_results/")
    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="./output/evaluation_result.jpg")

    args = parser.parse_args()

    return args

