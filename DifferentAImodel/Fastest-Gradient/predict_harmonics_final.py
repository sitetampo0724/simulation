
import numpy as np
import joblib
import warnings
from scipy import signal
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class HarmonicPredictor:
    """谐波预测器"""

    def __init__(self, model_path='square_wave_model.pkl',
                 features_path='square_wave_model_features.pkl',
                 labels_path='square_wave_model_labels.pkl'):
        print("=" * 60)
        print("=== Ideal Square Wave Harmonic Predictor ===")
        print("=" * 60)

        self.pipeline = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)
        self.label_names = joblib.load(labels_path)
        print(f"[OK] Model: {len(self.feature_names)} features, {len(self.label_names)} labels")

    def predict(self, freq, amplitude, duty_cycle=0.5, sampling_points=1000, duration=1.0):
        """Predict harmonic amplitudes"""
        from feature_extractor import extract_features_for_prediction

        print(f"\nInput: freq={freq}Hz, amplitude={amplitude}")


        T = 1 / freq
        t = np.linspace(0, T * duration, int(freq * sampling_points * duration), endpoint=False)
        square_wave = amplitude * signal.square(2 * np.pi * freq * t, duty=duty_cycle)


        fv, _ = extract_features_for_prediction(square_wave, freq, amplitude, duty_cycle, sampling_points)
        y_pred = self.pipeline.predict(fv.reshape(1, -1))[0]


        predictions = {}
        for i, label in enumerate(self.label_names):
            order = int(label.split('_')[1])
            predictions[order] = {'amp': y_pred[i]}

        self._display(predictions, freq, amplitude)
        return predictions, square_wave, t

    def _display(self, predictions, freq, amplitude):
        print(f"\n{'='*55}")
        print(f"Prediction Results (fundamental {freq:.1f} Hz)")
        print(f"{'Harmonic':>8} {'Freq(Hz)':>10} {'Predicted':>12} {'Theoretical':>12} {'Error%':>8}")
        print("-" * 55)
        for order in sorted(predictions.keys()):
            pred = predictions[order]['amp']
            theory = 4.0 * amplitude / (order * np.pi)
            err = abs(pred - theory) / theory * 100
            mk = " OK" if err < 1 else " ~" if err < 3 else " !"
            print(f"{order:6d}th {order*freq:9.1f} {pred:11.6f} {theory:11.6f} {err:7.3f}%{mk}")
        print(f"{'='*55}")

    def reconstruct(self, predictions, t, freq):
        """
        Reconstruct signal using sin (matches square wave FFT phase of -pi/2)
        f(t) = sum(4A/(n*pi) * sin(2*pi*n*f*t))
        """
        rec = np.zeros_like(t)
        for order in sorted(predictions.keys()):
            rec += predictions[order]['amp'] * np.sin(2 * np.pi * order * freq * t)
        return rec

    def plot(self, t, original, predictions, freq):
        """Plot reconstruction comparison + harmonic spectrum"""
        reconstructed = self.reconstruct(predictions, t, freq)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))


        ax = axes[0]
        ax.plot(t, original, 'b-', lw=2.5, label='Original Square Wave', alpha=0.8)
        ax.plot(t, reconstructed, 'r--', lw=1.5, label='Predicted Reconstruction')
        ax.set_title('Original vs Reconstruction (2 periods)', fontsize=12)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3, 3)

        ax = axes[1]
        orders = sorted(predictions.keys())
        pred_amps = [predictions[o]['amp'] for o in orders]
        freqs = [o * freq for o in orders]
        for f, a, o in zip(freqs, pred_amps, orders):
            c = '#D62828' if o == 1 else '#2E86AB'
            m, sl, bl = ax.stem([f], [a], basefmt=" ")
            plt.setp(m, marker='o', markersize=10, color=c)
            plt.setp(sl, linewidth=2, color=c)
            ax.annotate(f'{o}f\n{a:.3f}', xy=(f, a), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)
        ax.set_title('Predicted Harmonic Amplitude Spectrum', fontsize=12)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
        plt.show()

        mae = np.mean(np.abs(original - reconstructed))
        print(f"\nReconstruction MAE: {mae:.6f}")
        print(f"Original range: [{np.min(original):.3f}, {np.max(original):.3f}]")
        print(f"Reconstructed range: [{np.min(reconstructed):.3f}, {np.max(reconstructed):.3f}]")


def main():
    predictor = HarmonicPredictor()

    test_cases = [
        {'freq': 50, 'amplitude': 1.0},
        {'freq': 50, 'amplitude': 2.0},
        {'freq': 100, 'amplitude': 1.5},
    ]

    for case in test_cases:
        pred, sig, t = predictor.predict(**case, duration=2.0)
        predictor.plot(t, sig, pred, case['freq'])


if __name__ == "__main__":
    main()
