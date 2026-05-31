"""
feature_extractor.py - 特征提取模块（修复版本）
关键修复：
  1. 消除数据泄漏（谐波幅度不再同时作为特征和标签）
  2. 标签改用理论值 4A/(n*pi)
  3. 统一 extract_features_for_prediction 和 extract_square_wave_features_optimized 的特征名称
  4. 添加关键输入参数作为特征
"""

import numpy as np
from scipy import stats


CORE_FEATURE_NAMES = [
    'input_freq',
    'input_amplitude',
    'input_duty_cycle',
    'signal_amplitude',
    'signal_rms',
    'rise_time_est',
    'spectral_centroid',
    'spectral_bandwidth',
    'THD',
    'harmonic_decay_rate',
]

# 理论谐波阶数
HARMONIC_ORDERS = [1, 3, 5, 7, 9, 11]


def extract_features_for_prediction(signal_data, freq, amplitude, duty_cycle=0.5, sampling_points=1000):
    """
    提取特征 - 用于预测时
    与 extract_square_wave_features_optimized 生成完全相同的特征集
    """
    features = {}
    N = len(signal_data)
    fs = freq * sampling_points


    features['input_freq'] = freq
    features['input_amplitude'] = amplitude
    features['input_duty_cycle'] = duty_cycle


    features['signal_amplitude'] = np.max(signal_data) - np.min(signal_data)
    features['signal_rms'] = np.sqrt(np.mean(signal_data ** 2))


    if len(signal_data) > 10:
        sig_norm = (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data) + 1e-10)
        rising = np.where(np.diff(sig_norm) > 0.5)[0]
        features['rise_time_est'] = float(len(rising) / sampling_points * 1000) if len(rising) > 0 else 0.0
    else:
        features['rise_time_est'] = 0.0


    fft_result = np.fft.fft(signal_data)
    frequencies = np.fft.fftfreq(N, 1 / fs)
    positive_idx = frequencies > 0
    positive_freq = frequencies[positive_idx]
    positive_fft = fft_result[positive_idx]

    magnitude_spectrum = np.abs(positive_fft) / N * 2
    if len(magnitude_spectrum) > 0:
        magnitude_spectrum[0] = magnitude_spectrum[0] / 2

    # 4. 频谱特征
    if np.sum(magnitude_spectrum) > 0:
        features['spectral_centroid'] = float(
            np.sum(positive_freq * magnitude_spectrum) / np.sum(magnitude_spectrum))
    else:
        features['spectral_centroid'] = 0.0

    if features['spectral_centroid'] > 0 and np.sum(magnitude_spectrum) > 0:
        features['spectral_bandwidth'] = float(np.sqrt(
            np.sum((positive_freq - features['spectral_centroid']) ** 2 * magnitude_spectrum) /
            np.sum(magnitude_spectrum)))
    else:
        features['spectral_bandwidth'] = 0.0


    detected_harmonics = []
    for harmonic in HARMONIC_ORDERS:
        expected_freq = freq * harmonic
        freq_diff = np.abs(positive_freq - expected_freq)
        closest_idx = np.argmin(freq_diff)
        if freq_diff[closest_idx] < freq * 0.2:
            detected_harmonics.append({
                'order': harmonic,
                'magnitude': magnitude_spectrum[closest_idx]
            })

    if len(detected_harmonics) > 0:
        fundamental_power = detected_harmonics[0]['magnitude'] ** 2
        harmonic_powers = [h['magnitude'] ** 2 for h in detected_harmonics if h['order'] > 1]
        total_harmonic_power = np.sum(harmonic_powers)

        features['THD'] = float(np.sqrt(total_harmonic_power / (fundamental_power + 1e-10))) \
            if fundamental_power > 0 else 0.0

        if len(detected_harmonics) > 1:
            harmonic_orders = [h['order'] for h in detected_harmonics if h['order'] > 1]
            harmonic_amps = [max(h['magnitude'], 1e-10) for h in detected_harmonics if h['order'] > 1]
            if len(harmonic_orders) >= 2:
                try:
                    coeffs = np.polyfit(np.log(harmonic_orders), np.log(harmonic_amps), 1)
                    features['harmonic_decay_rate'] = float(-coeffs[0])
                except:
                    features['harmonic_decay_rate'] = 0.0
            else:
                features['harmonic_decay_rate'] = 0.0
        else:
            features['harmonic_decay_rate'] = 0.0
    else:
        features['THD'] = 0.0
        features['harmonic_decay_rate'] = 0.0


    feature_vector = np.array([features[name] for name in CORE_FEATURE_NAMES])
    return feature_vector, features


def extract_square_wave_features_optimized(square_wave, n_samples, sampling_points, n_harmonics=7):
    """
    从数据集提取特征和理论谐波标签
    与 extract_features_for_prediction 使用完全相同的特征提取逻辑
    """
    data = []
    labels_data = []

    for i in range(n_samples):
        sample = square_wave[i]
        signal_data = sample['signal']
        freq = sample['freq']
        amplitude = sample['amplitude']
        duty_cycle = 0.5


        feature_vector, _ = extract_features_for_prediction(
            signal_data, freq, amplitude, duty_cycle, sampling_points
        )
        data.append(feature_vector)


        labels = [4.0 * amplitude / (n * np.pi) for n in HARMONIC_ORDERS]
        labels_data.append(labels)

        if i < 3:
            print(f"\n样本 {i+1}: freq={freq:.1f}Hz, amp={amplitude:.4f}")
            print(f"  理论谐波: " + ", ".join(
                [f"{n}次:{4*amplitude/(n*np.pi):.4f}" for n in HARMONIC_ORDERS]
            ))

    data_array = np.array(data)
    labels_array = np.array(labels_data)
    label_names = [f'harmonic_{n}_amp' for n in HARMONIC_ORDERS]

    print(f"\n{'='*50}")
    print(f"特征提取完成: {data_array.shape}")
    print(f"标签: {label_names}")
    print(f"{'='*50}")

    return data_array, labels_array, CORE_FEATURE_NAMES, label_names
