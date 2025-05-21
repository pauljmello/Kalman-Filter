import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import matplotlib.pyplot as plt
from kalmanfilter import KalmanFilter


def generate_test_signal(t, amplitude, frequency, damping, noise_std):
    """Generate damped oscillations with noise"""
    clean = amplitude * np.exp(-damping * t) * np.cos(2 * np.pi * frequency * t)
    noisy = clean + np.random.normal(0, noise_std, len(t))
    return noisy, clean


def create_scenarios(time_config, scenario_configs):
    t = np.linspace(time_config['start'], time_config['end'], time_config['points'])

    scenarios = {}
    for name, config in scenario_configs.items():
        scenarios[name] = generate_test_signal(t, config['amplitude'], config['frequency'], config['damping'], config['noise_std'])

    if 'pulse_noise' in scenarios:
        signal, clean = scenarios['pulse_noise']
        impulse_idx = np.random.choice(len(signal), 20, replace=False)
        signal[impulse_idx] += np.random.normal(0, 0.8, 20)
        scenarios['pulse_noise'] = (signal, clean)

    return scenarios, t


def prepare_data(signal, seq_len=100):
    sequences = []
    for i in range(len(signal) - seq_len):
        sequences.append(signal[i:i + seq_len])
    return torch.tensor(np.array(sequences)[:, :, np.newaxis], dtype=torch.float32)


def run_kalman_filter(signal_tensor, kf_config):
    """Apply Kalman filter"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kf_model = KalmanFilter.create(kf_config['dt'], kf_config['damping'], kf_config['frequency'], kf_config['process_noise'],
                                   kf_config['observation_noise'], kf_config['state_dim'], kf_config['obs_dim'], device=device)
    signal_tensor = signal_tensor.to(device)

    kf_model.eval()
    with torch.no_grad():
        predictions = kf_model(signal_tensor)

    return predictions.cpu().numpy()


def plot_results(scenarios, time, predictions, save_dir='images'):
    """Generate plots"""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    titles = ['Low Noise', 'High Noise', 'Pulse Noise']

    for i, (name, (signal, clean)) in enumerate(scenarios.items()):
        pred = predictions[name][0, :, 0]
        seq_len = len(pred)
        t_sub = time[:seq_len]

        axes[i].plot(t_sub, clean[:seq_len], 'k-', label='True', linewidth=2)
        axes[i].plot(t_sub, signal[:seq_len], 'gray', label='Observed', alpha=0.85)
        axes[i].plot(t_sub, pred, 'r--', label='Kalman Filter', linewidth=2)

        mse = np.mean((pred - clean[:seq_len]) ** 2)
        axes[i].set_title(f'{titles[i]} (MSE: {mse:.4f})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/kalman_results.png', bbox_inches='tight')
    plt.close()

    print(f"Results saved to {save_dir}/kalman_results.png")


def main():
    # Config
    seed = 1001
    np.random.seed(seed)
    torch.manual_seed(seed)
    save_dir = 'images'

    # Data / Time Parameters
    time_config = {'start': 0, 'end': 50, 'points': 2500}

    # Kalman Filter Parameters
    kf_config = {'dt': 0.02, 'damping': 0.1, 'frequency': 1.0, 'process_noise': 0.01, 'observation_noise': 0.05, 'state_dim': 3, 'obs_dim': 1}

    # Kalman Filter Test Parameters
    scenario_configs = {'low_noise': {'amplitude': 2.0, 'frequency': 1.0, 'damping': 0.1, 'noise_std': 0.1},
                        'high_noise': {'amplitude': 1.5, 'frequency': 0.8, 'damping': 0.15, 'noise_std': 0.3},
                        'pulse_noise': {'amplitude': 1.8, 'frequency': 1.2, 'damping': 0.08, 'noise_std': 0.05}
    }

    print("Starting Kalman Filter")

    # Generate data
    scenarios, time = create_scenarios(time_config, scenario_configs)

    predictions = {}
    for name, (signal, clean) in scenarios.items():
        signal_tensor = prepare_data(signal)
        pred = run_kalman_filter(signal_tensor, kf_config)
        predictions[name] = pred

        mse = np.mean((pred[0, :, 0] - clean[:pred.shape[1]]) ** 2)
        print(f"{name}: MSE = {mse:.4f}")

    # Visualize
    plot_results(scenarios, time, predictions, save_dir)

if __name__ == "__main__":
    main()