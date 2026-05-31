import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import joblib
import warnings

warnings.filterwarnings('ignore')


def generate_training_data(n_samples=2000):
    """
    生成训练数据 - 使用与预测时完全一致的信号生成方式
    """
    from feature_extractor import extract_features_for_prediction, HARMONIC_ORDERS

    print("=" * 60)
    print(f"生成训练数据: {n_samples} 个样本")
    print("=" * 60)

    all_features = []
    all_labels = []

    freqs = np.random.uniform(10, 200, n_samples)
    amplitudes = np.random.uniform(0.5, 2.0, n_samples)

    for i, (freq, amp) in enumerate(zip(freqs, amplitudes)):
        if i % 500 == 0:
            print(f"  进度: {i}/{n_samples}")


        T = 1 / freq
        fs = freq * 1000
        duration = 1.0
        t = np.linspace(0, T * duration, int(fs * duration), endpoint=False)
        square_wave = amp * signal.square(2 * np.pi * freq * t, duty=0.5)


        feature_vector, _ = extract_features_for_prediction(
            square_wave, freq, amp, 0.5, 1000
        )
        all_features.append(feature_vector)


        labels = [4.0 * amp / (n * np.pi) for n in HARMONIC_ORDERS]
        all_labels.append(labels)

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\n数据形状: X={X.shape}, y={y.shape}")
    print("=" * 60)
    return X, y


def train_and_evaluate_model(X, y, feature_names, label_names, test_size=0.2):
    """训练并评估模型"""
    print("\n" + "=" * 60)
    print("训练模型...")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                min_samples_split=3,
                min_samples_leaf=2,
                subsample=0.9,
            ),
            n_jobs=-1
        ))
    ])


    print("\n交叉验证中...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    print(f"CV R2: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")


    print("\n训练中...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"评估结果:")
    print(f"  R2:  {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")


    perm = permutation_importance(pipeline, X_test, y_test, n_repeats=10,
                                   random_state=42, n_jobs=-1)
    print(f"\n特征重要性:")
    for idx in perm.importances_mean.argsort()[::-1]:
        print(f"  {feature_names[idx]:25s}: {perm.importances_mean[idx]:.4f}")

    return pipeline, X_test, y_test, y_pred


def save_model(pipeline, feature_names, label_names, path='square_wave_model'):
    """保存模型"""
    joblib.dump(pipeline, f'{path}.pkl')
    joblib.dump(feature_names, f'{path}_features.pkl')
    joblib.dump(label_names, f'{path}_labels.pkl')
    print(f"\n模型已保存: {path}.pkl")
