#!/usr/bin/env python3

import os
import time
import argparse
import torch as th
import torch.nn.functional as F
import numpy as np
from lext_tfgridnet import LExtTFGridNet
from libs.utils import load_json, get_logger
from libs.audio import write_wav
from libs.dataset_tse import Dataset


def chunk_inference(nnet, mix, aux, sr):
    """
    整句推理逻辑：确保输入为 2D [Batch, Time]
    """
    nnet.eval()
    device = mix.device

    # 确保输入至少是 2D (BatchSize=1)
    if mix.dim() == 1:
        mix = mix.unsqueeze(0)
    if aux.dim() == 1:
        aux = aux.unsqueeze(0)

    enroll_len = int(5.0 * sr)
    if aux.shape[-1] > enroll_len:
        aux_input = aux[..., :enroll_len]
    else:
        # 这里原代码用的是 F.pad，建议也改为循环填充，效果通常比补零好
        repeat_num = (enroll_len // aux.shape[-1]) + 1
        aux_input = aux.repeat(1, repeat_num)[..., :enroll_len]

    # 2. 直接进行全量推理
    # LExtTFGridNet 内部会处理 [e_norm ; glue ; m_norm] 的拼接
    with th.no_grad():
        final = nnet(mix, aux_input)

    return final

                                                                                                                                                
def run(args):
    start = time.time()
    logger = get_logger(
            os.path.join(args.checkpoint, 'separate.log'), file=True)
    dataset = Dataset(mix_scp=args.mix_scp, aux_scp=args.aux_scp, sample_rate=args.fs)
    
    # Load model
    nnet_conf = load_json(args.checkpoint, "mdl.json")
    nnet = LExtTFGridNet(**nnet_conf)
    cpt_fname = os.path.join(args.checkpoint, "best.pt.tar")
    cpt = th.load(cpt_fname, map_location="cpu")
    nnet.load_state_dict(cpt["model_state_dict"]) 
    logger.info("Load checkpoint from {}, epoch {:d}".format(
        cpt_fname, cpt["epoch"]))
    
    device = th.device(
        "cuda:{}".format(args.gpuid)) if args.gpuid >= 0 else th.device("cpu")
    nnet = nnet.to(device) if args.gpuid >= 0 else nnet
    nnet.eval()

    with th.no_grad():
        total_cnt = 0
        for i, data in enumerate(dataset):
            # 将 numpy 转换为 tensor
            key=data['key']
            mix_tensor = th.tensor(data['mix'], dtype=th.float32, device=device).unsqueeze(0)
            aux_tensor = th.tensor(data['aux'], dtype=th.float32, device=device).unsqueeze(0)
            mix_np=data['mix']

            target_seconds = 4.0
            enroll_len = int(target_seconds * 16000)
            if aux_tensor.shape[-1] > enroll_len:
                aux_tensor = aux_tensor[:, :enroll_len]
            elif aux_tensor.shape[-1] < enroll_len:
                # 方案：循环填充而非 F.pad
                repeat_num = (enroll_len // aux_tensor.shape[-1]) + 1
                aux_tensor = aux_tensor.repeat(1, repeat_num)[:, :enroll_len]

            ests_tensor = nnet(mix_tensor, aux_tensor)
            ests = ests_tensor.squeeze().cpu().numpy()
            # 3. 长度对齐 (防止 STFT 变换产生的微小点数差异)
            target_len = mix_np.shape[-1]
            if ests.shape[-1] > target_len:
                ests = ests[:target_len]
            elif ests.shape[-1] < target_len:
                ests = np.pad(ests, (0, target_len - ests.shape[-1]), "constant")
            est_max = np.max(np.abs(ests))
            mix_np = mix_tensor.squeeze(0).cpu().numpy()
            norm = np.linalg.norm(mix_np, np.inf)

            energy_threshold = 1e-4  # 静音阈值
            if est_max > energy_threshold:
                scale = (norm / est_max) * 0.9  # 算出缩放比例
                ests = ests * scale  # 强行把最大值拉到 0.9

            ests = np.clip(ests, -0.99, 0.99)

            logger.info("Separate Utt{:d}".format(total_cnt + 1))

            write_wav(os.path.join(args.dump_dir, key), ests, fs=args.fs)
            total_cnt += 1
    
    end = time.time()
    logger.info('Utt={:d} | Time Elapsed: {:.1f}s'.format(total_cnt, end-start))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Separating speech...')
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Directory of checkpoint")
    parser.add_argument("--gpuid", type=int, default=-1, 
                        help="GPU device to offload model to, -1 means running on CPU")
    parser.add_argument('--mix_scp', type=str, required=True,
                        help='mix scp')
    parser.add_argument('--ref_scp', type=str, required=False,
                        help='ref scp')
    parser.add_argument('--aux_scp', type=str, required=True,
                        help='aux scp')
    parser.add_argument('--fs', type=int, default=16000, 
                        help="Sample rate for mixture input")
    parser.add_argument('--dump-dir', type=str, default="/node/yym/expriment/LExt/result",
                        help="Directory to dump separated results out")
    args = parser.parse_args()
    run(args)
