import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
import random


def generate_scope_noise(t, snr_db, noise_type='mixed'):
    """生成示波器风格的高频密集噪声"""
    n_points = len(t)
    fs = 1 / (t[1] - t[0])

    # 1. 高频白噪声（宽带热噪声）- 示波器主要噪声源
    white_noise = np.random.randn(n_points)
    # 增强高频成分，让波形更密集
    b_hp, a_hp = signal.butter(4, 0.05, 'high')
    white_noise_hf = signal.filtfilt(b_hp, a_hp, white_noise)
    # 混合原始和高频增强版本
    white_noise = white_noise * 0.6 + white_noise_hf * 0.4

    # 2. 密集毛刺噪声（模拟接触不良、开关瞬态）
    glitches = np.zeros(n_points)
    # 高频毛刺 - 短持续时间、高幅度
    glitch_prob = 0.15  # 15%概率出现毛刺
    glitch_positions = np.random.random(n_points) < glitch_prob
    for pos in np.where(glitch_positions)[0]:
        duration = np.random.randint(1, 5)  # 1-4个采样点
        if pos + duration < n_points:
            amplitude = np.random.uniform(2, 6) * random.choice([-1, 1])
            glitches[pos:pos + duration] = amplitude

    # 3. 突发噪声块（模拟干扰脉冲）
    burst_noise = np.zeros(n_points)
    n_bursts = np.random.randint(2, 6)
    for _ in range(n_bursts):
        length = np.random.randint(10, 50)
        start = np.random.randint(0, n_points - length)
        burst_noise[start:start + length] = np.random.randn(length) * np.random.uniform(0.5, 2)

    # 4. 工频干扰（50Hz + 谐波）- 模拟电源耦合
    mains_noise = (0.3 * np.sin(2 * np.pi * 50 * t) +
                   0.15 * np.sin(2 * np.pi * 150 * t) +
                   0.08 * np.sin(2 * np.pi * 250 * t))

    # 5. 随机游走噪声（低频漂移）
    random_walk = np.cumsum(np.random.randn(n_points) * 0.02)
    random_walk = random_walk - np.mean(random_walk)

    # 6. 量化/台阶噪声（模拟ADC离散化）
    quant_step = 0.02
    quant_noise = np.random.uniform(-quant_step, quant_step, n_points)

    # 组合噪声（示波器风格：以高频毛刺+白噪声为主）
    if noise_type == 'mixed':
        raw_noise = (white_noise * 0.35 +
                     glitches * 0.30 +
                     burst_noise * 0.15 +
                     mains_noise * 0.10 +
                     random_walk * 0.07 +
                     quant_noise * 0.03)
    elif noise_type == 'white':
        raw_noise = white_noise
    elif noise_type == 'glitch':
        raw_noise = glitches * 2.0
    elif noise_type == 'burst':
        raw_noise = burst_noise * 1.5
    elif noise_type == 'mains':
        raw_noise = mains_noise
    else:
        raw_noise = white_noise

    # 根据信噪比调整幅度
    signal_power = 1
    noise_power_desired = signal_power / (10 ** (snr_db / 10))
    current_noise_power = np.mean(raw_noise ** 2)
    if current_noise_power > 0:
        scaling_factor = np.sqrt(noise_power_desired / current_noise_power)
        noise = raw_noise * scaling_factor
    else:
        noise = raw_noise

    return noise


def generate_noisy_square(t, freq, duty, amp, snr_db, noise_type='mixed'):
    """生成带示波器风格噪声的方波"""
    # 生成理想方波
    square = amp * signal.square(2 * np.pi * freq * t, duty=duty / 100)

    # 生成示波器风格噪声
    noise = generate_scope_noise(t, snr_db, noise_type)

    return square + noise, square


# ============ 主程序 ============

# 获取py文件所在目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 设置数据集目录
dataset_dir = os.path.join(parent_dir, 'NoiseDataDataset')
graph_dir = os.path.join(dataset_dir, 'Graph')
os.makedirs(graph_dir, exist_ok=True)

# 参数设置
fs = 5000             # 采样率提高到5000Hz，让波形更密集
duration = 0.3        # 每信号时长(秒)
n_points = int(fs * duration)

# 定义5个SNR等级及其参数
snr_levels = {
    'Very Clean': {'range': (20, 30), 'count': 200, 'color': '#00FF00', 'noise_type': 'white'},
    'Clean': {'range': (10, 20), 'count': 200, 'color': '#66FF66', 'noise_type': 'mixed'},
    'Moderate': {'range': (0, 10), 'count': 200, 'color': '#FFFF00', 'noise_type': 'mixed'},
    'Noisy': {'range': (-10, 0), 'count': 200, 'color': '#FFAA00', 'noise_type': 'mixed'},
    'Very Noisy': {'range': (-20, -10), 'count': 200, 'color': '#FF4444', 'noise_type': 'glitch'}
}

# 设置随机种子保证可重复
np.random.seed(42)
random.seed(42)

# 准备存储所有参数
all_freqs = []
all_snrs = []
all_duties = []
all_amps = []
all_noise_types = []
all_levels = []

# 按顺序生成每个等级的数据
sample_counter = 0
csv_data = None

for level_name, level_config in snr_levels.items():
    n_samples = level_config['count']
    snr_low, snr_high = level_config['range']
    noise_type = level_config['noise_type']

    print(f"生成 {level_name}: {n_samples} 张图片 (SNR: {snr_low}-{snr_high}dB)")

    # 为该等级生成参数
    freqs = np.random.uniform(3, 25, n_samples)
    snrs = np.random.uniform(snr_low, snr_high, n_samples)
    duties = np.random.uniform(15, 85, n_samples)
    amps = np.random.uniform(0.5, 3.0, n_samples)

    # 时间轴
    t = np.linspace(0, duration, n_points, endpoint=False)

    for i in range(n_samples):
        # 生成信号
        noisy_signal, clean_signal = generate_noisy_square(
            t, freqs[i], duties[i], amps[i], snrs[i], noise_type
        )

        # 存储数据
        if csv_data is None:
            csv_data = np.zeros((1000, n_points))
        csv_data[sample_counter, :] = noisy_signal

        # 保存参数
        all_freqs.append(freqs[i])
        all_snrs.append(snrs[i])
        all_duties.append(duties[i])
        all_amps.append(amps[i])
        all_noise_types.append(noise_type)
        all_levels.append(level_name)

        # ===== 示波器风格绘图 =====
        fig, ax = plt.subplots(figsize=(8, 3))

        # 黑色背景
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0a0a0a')

        # 计算显示范围 - 显示3个周期
        periods_to_show = min(3, int(duration * freqs[i]))
        if periods_to_show > 0:
            xlim_max = periods_to_show / freqs[i]
        else:
            xlim_max = duration

        # 绘制理想方波（暗绿色细线）
        ax.plot(t, clean_signal, color='#1a5c1a', linewidth=0.8,
                alpha=0.4, label='Ideal')

        # 绘制含噪声的波形（示波器荧光黄绿色）
        ax.plot(t, noisy_signal, color=level_config['color'], linewidth=0.7,
                alpha=0.95, label='Measured')

        # 坐标轴样式（示波器风格）
        ax.set_xlabel('Time (s)', fontsize=8, color='#888888')
        ax.set_ylabel('Voltage (V)', fontsize=8, color='#888888')
        ax.set_title(
            f'{sample_counter + 1:04d}: {level_name} | f={freqs[i]:.1f}Hz | duty={duties[i]:.0f}% | SNR={snrs[i]:.0f}dB',
            fontsize=9, color=level_config['color'])

        # 图例
        ax.legend(loc='upper right', fontsize=7,
                  facecolor='#1a1a1a', edgecolor='#444444',
                  labelcolor='#cccccc')

        # 网格线（示波器暗色网格）
        ax.grid(True, alpha=0.15, linestyle='--', color='#555555')

        # 范围设置
        ax.set_xlim(0, xlim_max)
        ax.set_ylim(-5, 5)

        # 刻度颜色
        ax.tick_params(colors='#888888', labelsize=7)

        # 边框颜色
        for spine in ax.spines.values():
            spine.set_color('#444444')

        plt.tight_layout()

        img_filename = os.path.join(graph_dir, f"{sample_counter + 1:04d}.png")
        plt.savefig(img_filename, dpi=100, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()

        sample_counter += 1

        if (i + 1) % 50 == 0:
            print(f"  已完成 {i + 1}/{n_samples}")

    print(f"  ✓ {level_name} 完成\n")

# 保存CSV
csv_filename = os.path.join(dataset_dir, "signals.csv")
np.savetxt(csv_filename, csv_data, delimiter=',', fmt='%.6f')

# 保存详细参数表
param_file = os.path.join(dataset_dir, "Statistics.csv")
import pandas as pd

df = pd.DataFrame({
    'sample_id': [f"{i + 1:04d}" for i in range(1000)],
    'freq_hz': all_freqs,
    'snr_db': all_snrs,
    'snr_level': all_levels,
    'duty_pct': all_duties,
    'amplitude_v': all_amps,
    'noise_type': all_noise_types
})

df.to_csv(param_file, index=False)

# 统计信息
print("=" * 50)
print("📊 数据集统计：")
print(f"  总样本数: 1000")
print("\n  SNR分布：")
for level in snr_levels.keys():
    count = df[df['snr_level'] == level].shape[0]
    print(f"    {level}: {count} 样本")
print(f"\n  噪声类型分布：")
print(df['noise_type'].value_counts())
print(f"\n  参数范围：")
print(f"    频率: {df['freq_hz'].min():.1f} - {df['freq_hz'].max():.1f} Hz")
print(f"    占空比: {df['duty_pct'].min():.0f} - {df['duty_pct'].max():.0f} %")
print(f"    幅度: {df['amplitude_v'].min():.1f} - {df['amplitude_v'].max():.1f} V")

csv_size = os.path.getsize(csv_filename) / 1024
print(f"\n✅ 完成！CSV文件大小: {csv_size:.1f} KB")
print(f"   数据集位置: {dataset_dir}")