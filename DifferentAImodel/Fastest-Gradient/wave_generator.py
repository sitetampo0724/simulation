
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 添加3D绘图支持

plt.rcParams['axes.unicode_minus'] = False


class Square_Wave_Generator:
    """方波信号生成器 - 生成理想和实际的方波信号"""

    def __init__(self, freq=10, amplitude=1.0, duty_cycle=0.5, sampling_points=1000, duration=1.0):
        self.freq = freq
        self.amplitude = amplitude
        self.duty_cycle = duty_cycle
        self.duration = duration
        self.sampling_points = sampling_points
        self.fs = None
        self.t = None

    def generate_ideal_square_wave(self, freq, amplitude, duty_cycle):
        """生成理想方波信号"""
        self.freq = freq
        T = 1 / self.freq
        self.fs = self.freq * self.sampling_points # 采样频率
        self.t = np.linspace(0, T * self.duration, int(self.sampling_points * self.duration), endpoint=False)
        ideal_square_wave = amplitude * signal.square(2 * np.pi * freq * self.t,
                                                           duty=self.duty_cycle)



        return ideal_square_wave

    def plot_waveform(self, signal_data, title="方波信号"):
        """绘制波形图"""
        plt.figure(figsize=(10, 4))
        plt.plot(self.t, signal_data)
        plt.title(title)
        plt.xlabel('时间 (s)')
        plt.ylabel('幅度')
        plt.grid(True)
        plt.show()

    def fourier_decomposition(self, signal_data, n_harmonics=10):
        """
        对信号进行傅里叶级数分解
        """
        N = len(signal_data)    # 信号长度（采样点数）

        # 执行FFT
        fft_result = np.fft.fft(signal_data)    # 执行FFT，得到复数频谱
        frequencies = np.fft.fftfreq(N, 1 / self.fs)    # 生成频率轴

        # 取单边频谱
        positive_freq_idx = frequencies > 0     # 筛选正频率索引
        positive_freq = frequencies[positive_freq_idx]      # 正频率部分
        positive_fft = fft_result[positive_freq_idx]         # 正频率对应的FFT结果

        # 计算幅度谱和相位谱
        magnitude_spectrum = np.abs(positive_fft) / N * 2  # 归一化    # 归一化幅度谱
        phase_spectrum = np.angle(positive_fft)     # 相位谱

        # 找到基频和主要谐波
        sorted_indices = np.argsort(magnitude_spectrum)[::-1]   # 按幅度降序排序
        fundamental_idx = sorted_indices[0]     # 最大幅度对应的频率索引
        fundamental_freq = positive_freq[fundamental_idx]   # 基频

        print("\n=== 傅里叶级数分解结果 ===")
        print(f"基频: {fundamental_freq:.2f} Hz")
        print(f"理论基频: {self.freq:.2f} Hz")
        print(f"采样率: {self.fs} Hz")
        print(f"信号长度: {N} 个采样点")
        print(f"持续时间: {1 / self.freq * self.duration:.3f} 秒")

        print(f"\n理想方波(50%占空比) - 前{n_harmonics}个奇次谐波分量:")
        harmonic_components = []    # 谐波信息
        harmonic_count = 0

        # 按幅度排序，但只选择奇次谐波
        for idx in sorted_indices:
            freq = positive_freq[idx]
            harmonic_ratio = freq / fundamental_freq

            # 检查是否为奇次谐波（允许小的容差）
            if abs(round(harmonic_ratio) - harmonic_ratio) < 0.1 and round(harmonic_ratio) % 2 == 1:
                magnitude = magnitude_spectrum[idx]
                phase = phase_spectrum[idx]

                harmonic_components.append({
                    'order': int(round(harmonic_ratio)),  # 谐波阶数
                    'frequency': freq,
                    'magnitude': magnitude,
                    'phase': phase,
                    'harmonic_ratio': harmonic_ratio    # 实际频率/基频
                })

                harmonic_count += 1
                if harmonic_count >= n_harmonics:
                    break

        print(f"\n前{n_harmonics}个主要谐波分量:")
        print("序号   谐波阶数    频率(Hz)    幅度      相位(rad)    与基频关系")
        print("-" * 55)

        for i, harmonic in enumerate(harmonic_components):
            print(f"{i + 1:2d}      {harmonic['order']:2d}       {harmonic['frequency']:8.2f}   "
                  f"{harmonic['magnitude']:8.4f}   {harmonic['phase']:8.4f}    "
                  f"{harmonic['harmonic_ratio']:5.1f}倍")

        # 重建信号（使用选定的谐波）
        reconstructed = np.zeros_like(signal_data, dtype=float)
        # 存储每个谐波分量的时域波形，用于3D绘图
        harmonic_waveforms = []     # 谐波分量时域波形

        for i, harmonic in enumerate(harmonic_components):
            # 重建每个谐波分量
            harmonic_wave = harmonic['magnitude'] * np.cos(
                2 * np.pi * harmonic['frequency'] * self.t + harmonic['phase']
            )
            harmonic_waveforms.append(harmonic_wave)
            reconstructed += harmonic_wave

        # 绘制结果，频谱图、相位谱图、及合成图
        # self.plot_decomposition_results(signal_data, positive_freq, magnitude_spectrum, phase_spectrum,
        #     reconstructed, harmonic_components, title_suffix="(测试信号)")

        # 3D绘图函数
        # self.plot_3d_signal_reconstruction(signal_data, reconstructed,
        #                                    harmonic_waveforms, harmonic_components,
        #                                    title_suffix="(测试信号)")

        return harmonic_components

    def plot_3d_signal_reconstruction(self, original_signal, reconstructed_signal, harmonic_waveforms,
                                      harmonic_components, title_suffix=""):
        """绘制信号重建的三维图，展示每个谐波分量的叠加过程"""
        # 创建3D图形
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 准备数据
        n_harmonics = len(harmonic_components)

        # 为每个谐波分量创建3D线图
        for i, (wave, harmonic) in enumerate(zip(harmonic_waveforms, harmonic_components)):
            # 每个谐波在z轴（谐波阶数）上偏移显示
            z_offset = harmonic['order']  # 使用谐波阶数作为z轴

            # 绘制谐波波形
            ax.plot(self.t, wave, z_offset,
                    label=f'谐波{harmonic["order"]} ({harmonic["frequency"]:.1f}Hz)',
                    alpha=0.8, linewidth=1.5)

        # 绘制原始信号（在顶部）
        max_harmonic_order = max([h['order'] for h in harmonic_components])
        ax.plot(self.t, original_signal, max_harmonic_order + 1,
                'r-', linewidth=2, label='原始信号')

        # 绘制重建信号（在原始信号下方）
        ax.plot(self.t, reconstructed_signal, max_harmonic_order + 1,
                'g--', linewidth=2, label=f'重建信号({n_harmonics}个谐波)')

        ax.set_xlabel('时间 (s)', fontsize=12, labelpad=10)
        ax.set_ylabel('幅度', fontsize=12, labelpad=10)
        ax.set_zlabel('谐波阶数', fontsize=12, labelpad=10)

        # 生成以2为间隔的刻度
        z_ticks = list(range(0, max_harmonic_order + 2, 2))
        ax.set_zticks(z_ticks)

        ax.set_title(f'{self.freq}Hz 方波谐波分量分解 - 3D视图 {title_suffix}', fontsize=14, pad=20)

        ax.view_init(elev=20, azim=-45)

        # 添加图例
        # ax.legend(loc='upper right', fontsize='small')
        ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.1, 1), borderaxespad=0)

        ax.grid(True, alpha=0.3)

        plt.tight_layout(pad=4.0)
        plt.show()

    def plot_decomposition_results(self, original_signal, frequencies, magnitudes, phases,
                                   reconstructed_signal, harmonics, title_suffix=""):
        """绘制傅里叶分解结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 14))

        # 原始信号
        axes[0, 0].plot(self.t, original_signal, 'b-', linewidth=1.5)
        axes[0, 0].set_title(f'原始方波信号 ({self.freq}Hz, 占空比:{self.duty_cycle * 100}%) {title_suffix}')
        axes[0, 0].set_xlabel('时间 (s)')
        axes[0, 0].set_ylabel('幅度')
        axes[0, 0].grid(True)
        axes[0, 0].set_xlim(0, (self.duration / self.freq) * 1.1)

        freq_range = None
        if freq_range is None:
            # 默认显示到第10次谐波频率
            max_harmonic_order = max([h['order'] for h in harmonics]) if harmonics else 10
            freq_range = (0, self.freq * max_harmonic_order * 1.1)

        # 幅度谱
        # 筛选在指定频率范围内的频率点
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        freq_subset = frequencies[freq_mask]
        mag_subset = magnitudes[freq_mask]
        phase_subset = phases[freq_mask]  # 同时获取相位子集

        # 限制显示的谐波数量，避免过于密集
        # max_harmonics_to_show = 20
        max_harmonics_to_show = len(harmonics)
        if len(freq_subset) > max_harmonics_to_show:
            # 只显示幅度最大的前20个
            top_indices = np.argsort(mag_subset)[-max_harmonics_to_show:]
            freq_subset = freq_subset[top_indices]
            mag_subset = mag_subset[top_indices]
            phase_subset = phase_subset[top_indices]  # 这里同步筛选相位

        markerline, stemlines, baseline = axes[0, 1].stem(
            freq_subset, mag_subset, basefmt=" "
        )
        plt.setp(markerline, marker='o', markersize=5, color='red')
        plt.setp(stemlines, linewidth=1.0, color='blue')
        axes[0, 1].set_title(f'幅度频谱{title_suffix}')
        axes[0, 1].set_xlabel('频率 (Hz)')
        axes[0, 1].set_ylabel('幅度')
        axes[0, 1].grid(True)
        axes[0, 1].set_xlim(freq_range)

        # 增加幅度谱的y轴上限，为标注留出空间
        y_max = mag_subset.max() if len(mag_subset) > 0 else 1.0
        axes[0, 1].set_ylim(0, y_max * 1.1)

        # 标记主要谐波 - 改进标注位置
        for i, harmonic in enumerate(harmonics):
            if freq_range[0] <= harmonic['frequency'] <= freq_range[1]:
                # 根据位置调整标注偏移
                x_offset = 5
                y_offset = 10 if i % 2 == 0 else -15  # 交替上下标注
                axes[0, 1].annotate(f'{harmonic["order"]}f',
                                    xy=(harmonic['frequency'], harmonic['magnitude']),
                                    xytext=(x_offset, y_offset),
                                    textcoords='offset points',
                                    fontsize=8,  # 减小字体大小
                                    fontweight='bold',
                                    arrowprops=dict(arrowstyle='-', lw=0.5))

        # 绘制相位谱
        markerline2, stemlines2, baseline2 = axes[1, 0].stem(
            freq_subset, phase_subset, basefmt=" "
        )
        plt.setp(markerline2, marker='o', markersize=5, color='red')
        plt.setp(stemlines2, linewidth=1.0, color='green')
        axes[1, 0].set_title(f'相位频谱{title_suffix}')
        axes[1, 0].set_xlabel('频率 (Hz)')
        axes[1, 0].set_ylabel('相位 (rad)')
        axes[1, 0].grid(True)
        axes[1, 0].set_xlim(freq_range)
        axes[1, 0].set_ylim(-np.pi, np.pi)  # 相位通常限制在[-π, π]范围内
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

        # 标记主要谐波的相位 - 改进标注
        for i, harmonic in enumerate(harmonics):
            if freq_range[0] <= harmonic['frequency'] <= freq_range[1]:
                y_offset = 10 if harmonic['phase'] > 0 else -15  # 根据相位值调整方向
                axes[1, 0].annotate(f'{harmonic["order"]}f',
                                    xy=(harmonic['frequency'], harmonic['phase']),
                                    xytext=(5, y_offset),
                                    textcoords='offset points',
                                    fontsize=8,
                                    fontweight='bold',
                                    arrowprops=dict(arrowstyle='-', lw=0.5))

        # 重建信号对比
        axes[1, 1].plot(self.t, original_signal, 'b-', label='原始信号', alpha=0.7, linewidth=2)
        axes[1, 1].plot(self.t, reconstructed_signal, 'r--',
                     label=f'重建信号({len(harmonics)}个谐波)', linewidth=1.5)

        # 计算重建误差
        error = np.mean(np.abs(original_signal - reconstructed_signal))
        axes[1, 1].set_title(f'信号重建对比 (平均误差: {error:.4f}) {title_suffix}')
        axes[1, 1].set_xlabel('时间 (s)')
        axes[1, 1].set_ylabel('幅度')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_xlim(0, (self.duration / self.freq) * 1.1)

        plt.tight_layout(pad=4.0, w_pad=3.0, h_pad=6.0)  # 增加子图间距
        plt.show()

