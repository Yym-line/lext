#!/usr/bin/env python3
import numpy as np
# 兼容旧版 mir_eval 调用 np.Inf
if not hasattr(np, "Inf"):
        np.Inf = np.inf

import os
import time
import argparse
import torch as th
import numpy as np
import torch.nn.functional as F
from mir_eval.separation import bss_eval_sources
from pesq import pesq as pesq2
from pypesq import pesq as pesq1
from pystoi.stoi import stoi
from lext_tfgridnet import LExtTFGridNet
from libs.utils import load_json, get_logger
from libs.dataset_tse import Dataset


def chunk_inference(nnet, mix, aux, sr=16000, chunk_len=3.0, hop_len=1.5):
    """
    mix: [1, T_mix] 全长混合信号
    aux: [1, T_aux] 辅助参考信号 (Enrollment)
    """
    T = mix.shape[-1]
    chunk_size = int(chunk_len * sr)
    hop_size = int(hop_len * sr)

    # 初始化输出缓存和权重统计
    out_total = th.zeros_like(mix)
    weight_total = th.zeros_like(mix)

    # 定义汉宁窗用于平滑拼接
    window = th.hann_window(chunk_size).to(mix.device)

    # 固定取 Enrollment 的前 2s (对应训练长度)
    enroll_limit = int(2.0 * sr)
    aux_fixed = aux[:, :enroll_limit] if aux.shape[-1] > enroll_limit else aux

    for start in range(0, T - chunk_size + 1, hop_size):
        end = start + chunk_size
        mix_chunk = mix[:, start:end]

        # 调用模型推理
        # 模型内部会处理 [aux_fixed ; glue ; mix_chunk]
        with th.no_grad():
            sep_chunk = nnet(mix_chunk, aux_fixed)

            # 累加结果
        out_total[:, start:end] += sep_chunk * window
        weight_total[:, start:end] += window

    # 处理最后一段不足 chunk_size 的部分
    if T > chunk_size and (T - chunk_size) % hop_size != 0:
        # 补齐最后一段逻辑... (略)
        pass

    # 归一化权重，防止拼接处音量翻倍
    return out_total / (weight_total + 1e-8)

def evaluate(args, model_file, logger):
    start = time.time()
    total_SISNR = 0
    total_SISNRi = 0
    total_PESQ = 0
    total_PESQi = 0
    total_PESQ2 = 0
    total_PESQi2 = 0
    total_STOI = 0
    total_STOIi = 0
    total_SDR = 0
    total_cnt = 0

    # Load model
    nnet_conf = load_json(args.checkpoint, "mdl.json")
    nnet = LExtTFGridNet(**nnet_conf)
    cpt_fname = os.path.join(args.checkpoint, model_file)
    cpt = th.load(cpt_fname, map_location="cpu")
    nnet.load_state_dict(cpt["model_state_dict"])
    logger.info("Loaded checkpoint from {}, epoch {:d}".format(
        cpt_fname, cpt["epoch"]))
    
    device = th.device(
        "cuda:{}".format(args.gpuid)) if args.gpuid >= 0 else th.device("cpu")
    nnet = nnet.to(device) if args.gpuid >= 0 else nnet
    nnet.eval()

    # Load data
    dataset = Dataset(mix_scp=args.mix_scp, ref_scp=args.ref_scp, aux_scp=args.aux_scp, sample_rate=16000)

    with th.no_grad():
        for i, data in enumerate(dataset):
            # 转换为 Tensor 并移动到设备
            mix_tensor = th.tensor(data['mix'], dtype=th.float32, device=device).unsqueeze(0)
            aux_tensor = th.tensor(data['aux'], dtype=th.float32, device=device).unsqueeze(0)

            enroll_len = int(2.0 * 16000)  # 训练时的 2s
            if aux_tensor.shape[-1] > enroll_len:
                aux_tensor = aux_tensor[:, :enroll_len]
            elif aux_tensor.shape[-1] < enroll_len:
                aux_tensor = F.pad(aux_tensor, (0, enroll_len - aux_tensor.shape[-1]))
            # 统一使用 chunk_inference (即便不分块，它内部的归一化也更鲁棒)
            # 或者如果你想整句跑，确保归一化逻辑与训练一致
            ests_tensor = nnet(mix_tensor,aux_tensor)
            ests_np = ests_tensor.squeeze().cpu().numpy()
            ref_np = data['ref']  # 原始 numpy
            mix_np = data['mix']

            # --- 关键：统一强制对齐 ---
            t_len = min(ests_np.shape[0], ref_np.shape[0])
            ests_np = ests_np[:t_len]
            ref_np = ref_np[:t_len]
            mix_np = mix_np[:t_len]

            # Compute metrics
            if args.cal_sdr == 1:
                SDR, sir, sar, popt = bss_eval_sources(ref_np, ests_np)
                total_SDR += SDR[0]
            SISNR, delta = cal_SISNRi(ests_np, ref_np, mix_np)
            PESQ, PESQi, PESQ2, PESQi2 = cal_PESQi(ests_np, ref_np, mix_np)
            STOI, STOIi = cal_STOIi(ests_np, ref_np, mix_np)
            if args.cal_sdr == 1:
                logger.info("Utt={:d} | SDR={:.2f} | SI-SNR={:.2f} | SI-SNRi={:.2f} | PESQ={:.2f} | PESQi={:.2f}| PESQ2={:.2f} | PESQi2={:.2f} | | STOI={:.2f} | STOIi={:.2f}".format(
                    total_cnt+1, SDR[0], SISNR, delta, PESQ, PESQi, PESQ2, PESQi2, STOI, STOIi))
            else:
                logger.info("Utt={:d} | SI-SNR={:.2f} | SI-SNRi={:.2f} | PESQ={:.2f} | PESQi={:.2f} | PESQ2={:.2f} | PESQi2={:.2f} | STOI={:.2f} | STOIi={:.2f}".format(
                    total_cnt+1, SISNR, delta, PESQ, PESQi, PESQ2, PESQi2, STOI, STOIi))
            total_SISNR += SISNR
            total_SISNRi += delta
            total_PESQ += PESQ
            total_PESQi += PESQi
            total_PESQ2 += PESQ2
            total_PESQi2 += PESQi2
            total_STOI += STOI
            total_STOIi += STOIi
            total_cnt += 1
    end = time.time()
    
    logger.info('Time Elapsed: {:.1f}s'.format(end-start))
    if args.cal_sdr == 1:
        logger.info("Average SDR: {0:.2f}".format(total_SDR / total_cnt))
    logger.info("Average SI-SNR: {:.2f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.2f}".format(total_SISNRi / total_cnt))
    logger.info("Average PESQ: {:.2f}".format(total_PESQ / total_cnt))
    logger.info("Average PESQi: {:.2f}".format(total_PESQi / total_cnt))
    logger.info("Average PESQ2: {:.2f}".format(total_PESQ2 / total_cnt))
    logger.info("Average PESQi2: {:.2f}".format(total_PESQi2 / total_cnt))
    logger.info("Average STOI: {:.2f}".format(total_STOI / total_cnt))
    logger.info("Average STOIi: {:.2f}".format(total_STOIi / total_cnt))

def cal_SISNR(est, ref, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """ 
    assert len(est) == len(ref)
    est_zm = est - np.mean(est)
    ref_zm = ref - np.mean(ref)

    t = np.sum(est_zm * ref_zm) * ref_zm / (np.linalg.norm(ref_zm)**2 + eps)
        
    return 20 * np.log10(eps + np.linalg.norm(t) / (np.linalg.norm(est_zm - t) + eps))

def cal_SISNRi(est, ref, mix, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """ 
    assert len(est) == len(ref) == len(mix)
    sisnr1 = cal_SISNR(est, ref)
    sisnr2 = cal_SISNR(mix, ref)
    
    return sisnr1, sisnr1 - sisnr2
                         
def cal_PESQ(est, ref):
    assert len(est) == len(ref)
    mode ='wb'
    p = pesq1(ref, est,16000)
    p2 = pesq2(16000, ref, est, mode)
    return p,p2

def cal_PESQi(est, ref, mix):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    pesq1,pesq12 = cal_PESQ(est, ref)
    pesq2,pesq22= cal_PESQ(mix, ref)

    return pesq1, pesq1 - pesq2,pesq12,pesq12-pesq22

def cal_STOI(est, ref):
    assert len(est) == len(ref)
    p = stoi(ref, est, 16000)
    return p

def cal_STOIi(est, ref, mix):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        est: separated signal, numpy.ndarray, [T]
        ref: reference signal, numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(est) == len(ref) == len(mix)
    stoi1 = cal_STOI(est, ref)*100
    stoi2 = cal_STOI(mix, ref)*100

    return stoi1, stoi1 - stoi2

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
    parser.add_argument('--checkpoint', type=str,
                        default='/node/yym/expriment/LExt/demo/',
                        help='Path to model directory containing checkpoints')
    parser.add_argument('--gpuid', type=int, default=0,
                        help="GPU device to offload model to, -1 means running on CPU")  
    parser.add_argument('--mix_scp', type=str,
                        default='/node/yym/node/LExt/data/test/mix_clean.scp',
                        help='mix scp')
    parser.add_argument('--ref_scp', type=str,
                        default='/node/yym/node/LExt/data/test/ref.scp',
                        help='ref scp')
    parser.add_argument('--aux_scp', type=str,
                        default='/node/yym/node/LExt/data/test/auxs1.scp',
                        help='aux scp')    
    parser.add_argument('--cal_sdr', type=int, default=None,
                        help='Whether calculate SDR, add this option because calculation of SDR is very slow')

    args = parser.parse_args()

    
    # eval best.pt.tar
    best_model_file = "best.pt.tar"
    best_log_file = os.path.join(args.checkpoint, "eval_best.log")
    best_logger = get_logger(best_log_file, file=True)
    best_logger.info(f"Evaluating model: {best_model_file}")
    evaluate(args, best_model_file, best_logger)
    
    # eval 110-122 epoch.pt.tar
    for epoch in range(110, 122):
        model_file = f"{epoch}.pt.tar"
        log_file = os.path.join(args.checkpoint, f"eval_{epoch}.log")
        logger = get_logger(log_file, file=True)
        logger.info(f"Evaluating model: {model_file}")
        evaluate(args, model_file, logger)
