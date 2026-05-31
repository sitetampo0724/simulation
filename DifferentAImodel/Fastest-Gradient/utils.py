import joblib
import pandas as pd
import numpy as np

def save_model(model_pipeline, feature_names, label_names, filepath='square_wave_model'):
    model_file = f'{filepath}.pkl'
    joblib.dump(model_pipeline, model_file)

    feature_file = f'{filepath}_features.pkl'
    joblib.dump(feature_names, feature_file)

    label_file = f'{filepath}_labels.pkl'
    joblib.dump(label_names, label_file)

    print(f"\n=== 保存训练模型 ===")
    print(f"模型已保存到: {model_file}")
    print(f"\n=== 特征模型 ===")
    print(f"特征名称已保存到: {feature_file}")
    print(f"\n=== 标签模型 ===")
    print(f"标签名称已保存到: {label_file}")

    return model_file, feature_file, label_file
