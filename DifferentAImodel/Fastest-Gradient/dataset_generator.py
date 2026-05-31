"""
模拟生成数据集
"""

import numpy as np
from wave_generator import Square_Wave_Generator
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False

def display_dataset_samples(dataset, n_samples):
    """
    显示数据集前n个样本的信息
    """
    print(f"\n=== 数据集前{n_samples}个样本的信息 ===")
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        print(f"样本 {sample['id'] + 1}:")
        print(f"  频率: {sample['freq']:.2f} Hz")
        print(f"  幅度: {sample['amplitude']:.2f}")
        print(f"  信号长度: {len(sample['signal'])} 个点")
        # print(f"  信号前10个值: {sample['signal'][:10]}")
        print("-" * 40)

def generate_dataset(n_samples, sampling_points, duration):
    print(f"正在生成 {n_samples} 个样本的数据集...")
    # 创建并初始化一个方波生成器对象
    generator = Square_Wave_Generator(freq=10,
                                      amplitude=1.0,
                                      duty_cycle=0.5,
                                      sampling_points=sampling_points,
                                      duration=duration)

    # 初始化存储所有样本的列表
    dataset = []

    for i in range(n_samples):
        if (i + 1) % 1000 == 0:
            print(f"  已生成 {i + 1}/{n_samples} 个样本")

        freq = int(np.random.uniform(10, 200))
        amplitude = np.random.uniform(0.5, 2.0)

        wave = generator.generate_ideal_square_wave(
            freq=freq,
            amplitude=amplitude,
            duty_cycle=0.5
        )

        # 将样本添加到数据集
        dataset.append({
            'id': i,
            'freq': freq,
            'amplitude': amplitude,
            'signal': wave
        })

    # 调用函数显示数据集前5个样本的信息
    #display_dataset_samples(dataset, n_samples=n_samples)
    print(f"\n样本数据集生成完毕！")

    return dataset
