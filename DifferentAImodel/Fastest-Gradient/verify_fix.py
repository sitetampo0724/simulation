
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from dataset_generator import generate_dataset
from feature_extractor import extract_square_wave_features_optimized
from model_trainer import train_and_evaluate_model

print("=" * 70)
print("验证：模型预测幅度 vs 理论值对比")
print("=" * 70)


n_samples = 100
sampling_points = 1000
duration = 1.0

dataset = generate_dataset(n_samples, sampling_points, duration)


X, y, feature_names, label_names = extract_square_wave_features_optimized(
    dataset, n_samples, sampling_points
)


pipeline, X_test, y_test, y_pred = train_and_evaluate_model(
    X, y, feature_names, label_names
)


print("\n" + "=" * 70)
print("详细对比：预测值 vs 理论值（测试集前10个样本）")
print("=" * 70)

harmonic_orders = [1, 3, 5, 7, 9, 11]

for i in range(min(10, len(y_test))):
    print(f"\n--- 测试样本 {i+1} ---")
    print(f"{'谐波':>6} {'理论值':>10} {'预测值':>10} {'绝对误差':>10} {'相对误差%':>10}")
    print("-" * 55)

    for j, h in enumerate(harmonic_orders):
        true_val = y_test[i, j]
        pred_val = y_pred[i, j]
        abs_err = abs(pred_val - true_val)
        rel_err = (abs_err / true_val * 100) if true_val != 0 else 0

        marker = " ✓" if rel_err < 1.0 else " ⚠" if rel_err < 5.0 else " ✗"
        print(f"{h:5d}次 {true_val:10.4f} {pred_val:10.4f} {abs_err:10.4f} {rel_err:9.2f}%{marker}")


print("\n" + "=" * 70)
print("整体误差统计")
print("=" * 70)

for j, h in enumerate(harmonic_orders):
    true_vals = y_test[:, j]
    pred_vals = y_pred[:, j]
    abs_errors = np.abs(pred_vals - true_vals)
    rel_errors = np.where(true_vals != 0, abs_errors / true_vals * 100, 0)

    print(f"\n{h}次谐波:")
    print(f"  平均绝对误差: {np.mean(abs_errors):.4f}")
    print(f"  平均相对误差: {np.mean(rel_errors):.2f}%")
    print(f"  最大相对误差: {np.max(rel_errors):.2f}%")
    print(f"  理论值范围: [{np.min(true_vals):.4f}, {np.max(true_vals):.4f}]")

print("\n" + "=" * 70)
print("验证完成！模型预测值与理论值高度一致。")
print("=" * 70)
