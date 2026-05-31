import warnings
warnings.filterwarnings('ignore')

from model_trainer import generate_training_data, train_and_evaluate_model, save_model
from feature_extractor import CORE_FEATURE_NAMES, HARMONIC_ORDERS


def main():
    print("=" * 70)
    print("=== 理想方波谐波分析模型训练 ===")
    print("=" * 70)

    n_samples = 2000


    X, y = generate_training_data(n_samples)


    label_names = [f'harmonic_{n}_amp' for n in HARMONIC_ORDERS]


    pipeline, X_test, y_test, y_pred = train_and_evaluate_model(
        X, y, CORE_FEATURE_NAMES, label_names
    )


    save_model(pipeline, CORE_FEATURE_NAMES, label_names)

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
