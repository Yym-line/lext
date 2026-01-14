import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tfgridnetv2_separator import TFGridNetV2
from memonger import SublinearSequential

def param(nnet, Mb=True):
    neles = sum([p.nelement() for p in nnet.parameters()])
    return neles / 10 ** 6 if Mb else neles

class LExtTFGridNet(nn.Module):
    def __init__(self, sample_rate=16000, use_memonger=True, **kwargs):
        """
        使用 **kwargs 接收 conf.py 中的所有 nnet_conf 参数
        """
        super(LExtTFGridNet, self).__init__()

        # 1. 自动计算 input_dim
        # TFGridNet 需要知道频点数：n_fft // 2 + 1
        n_fft = kwargs.get("n_fft", 256)
        if "input_dim" not in kwargs:
            kwargs["input_dim"] = n_fft // 2 + 1

        # 强制设置 n_imics=1，因为 LExt 是单通道时间轴拼接逻辑
        kwargs["n_imics"] = 1

        # 2. 初始化基础网络 TFGridNetV2
        # 将所有配置参数透传进去
        self.nnet = TFGridNetV2(**kwargs)

        # 2. 显存优化：使用 memonger 包装 TFGridNet 内部的 Block 层
        if use_memonger:
            # TFGridNetV2 内部的 blocks 是 nn.ModuleList
            # 我们将其转换为 SublinearSequential 以通过重算节省内存
            self.nnet.blocks = SublinearSequential(*self.nnet.blocks)
            print("Successfully wrapped TFGridNet blocks with SublinearSequential.")

        # 3. LExt 专用参数
        self.sample_rate = sample_rate
        self.glue_len = int(sample_rate * 0.032)  # 论文定义的 32ms 胶水信号长度
        self.glue_val = 5.0  # 论文推荐的胶水信号数值

    def forward(self,
                mix: th.Tensor,
                enrollment: th.Tensor) -> th.Tensor:

        # A. 增益归一化 (Gain Normalization)

        std_m = torch.std(mix, dim=-1, keepdim=True) + 1e-8
        std_e = torch.std(enrollment, dim=-1, keepdim=True) + 1e-8
        m_norm = mix / std_m
        e_norm = enrollment / std_e


        # B. 构造 LExt 输入信号 [Enrollment ; Glue ; Mixture]
        batch_size = mix.shape[0]
        device = mix.device

        # 创建胶水信号 (全 5.0 填充)
        glue = torch.full((batch_size, self.glue_len), self.glue_val, device=device)

        # 时间轴拼接 (Dimension 1)
        input_wave = torch.cat([e_norm, glue, m_norm], dim=1)

        # C. 模型推理
        ilens = torch.full((batch_size,), input_wave.shape[1], device=device, dtype=torch.long)

        # 调用 TFGridNetV2
        outputs, _, _ = self.nnet(input_wave, ilens)

        # D. 输出裁剪与音量恢复
        cut_offset = enrollment.shape[1] + self.glue_len

        # 提取目标部分 [B, T_mix]
        est_target = outputs[0][:, cut_offset:]

        # 鲁棒性处理：确保长度与原始 mixture 严格对齐 (防止 STFT 补零偏差)
        if est_target.shape[1] > mix.shape[1]:
            est_target = est_target[:, :mix.shape[1]]
        elif est_target.shape[1] < mix.shape[1]:
            est_target = F.pad(est_target, (0, mix.shape[1] - est_target.shape[1]))

        if self.training:
            return est_target
        else:
            return est_target * std_m

# 测试代码
if __name__ == "__main__":
    from thop import profile, clever_format
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # 模拟 nnet_conf 配置
    conf = {
        "n_srcs": 1,
        "n_fft": 256,             # 16ms 窗长
        "stride": 128,            # 8ms 步长
        "n_layers": 4,            # TFGridNet 层数
        "lstm_hidden_units": 128,
        "emb_dim": 64,
        "emb_ks": 4,
        "emb_hs": 4,
        "sample_rate": 16000,
        "use_memonger": True,
        "window": "hann"
    }
    model = LExtTFGridNet(**conf)
    print(f"Model initialized. Parameters: {param(model)} Mb")

    # 模拟输入 [Batch=2, Time]
    mix = torch.randn(2, 48000)
    aux = torch.randn(2, 32000)

    model.eval()

    flops = FlopCountAnalysis(model, (mix, aux))
    macs = flops.total() / 2
    print("MACs:", macs)

    print("FLOPs:", flops.total())
    print(parameter_count_table(model))
    #out = model(mix, aux)
    #macs, params = profile(model, inputs=(mix, aux),verbose=True)
    #macs, params = clever_format([macs, params], "%.3f")
    #print(macs, params)
    #print(f"Input mix: {mix.shape}, Output: {out.shape}")