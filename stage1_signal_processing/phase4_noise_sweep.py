"""
Phase IV (extended) -- Noise-Robustness Sweep
=================================================
The clean-accuracy tournament in phase4_tournament.py found the from-scratch
CNN slightly AHEAD of classical ML -- which does not, on its own, contradict
Paredes Ocaranza et al. (2025) [12], because clean same-distribution accuracy
is not the condition their study found the largest gap under. This script
tests the actual condition: injected consumer-grade noise, swept across
increasing severity, exactly mirroring their sigma sweep.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from phase4_tournament import (
    simulate_session, extract_features, train_cnn, predict_cnn, FS
)


def evaluate_at_noise_level(sigma, n_sessions=50, seed=7):
    rng = np.random.default_rng(seed)
    true_focus = rng.uniform(0, 1, n_sessions)
    raw_clean = np.array([simulate_session(f, seed=i) for i, f in enumerate(true_focus)])
    # Inject additional Gaussian noise on top of the existing generative noise,
    # at increasing severity -- the same style of perturbation the cited study used.
    noise = rng.standard_normal(raw_clean.shape) * sigma
    raw_noisy = raw_clean + noise
    X_feat = np.array([extract_features(w) for w in raw_noisy])
    self_report = np.clip(np.round(1 + true_focus * 4 + rng.normal(0, 0.4, n_sessions)), 1, 5)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    preds_rf = np.zeros(n_sessions)
    preds_cnn = np.zeros(n_sessions)

    for train_idx, test_idx in kf.split(X_feat):
        rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=0)
        rf.fit(X_feat[train_idx], self_report[train_idx])
        preds_rf[test_idx] = rf.predict(X_feat[test_idx])

        net, y_mean, y_std = train_cnn(raw_noisy[train_idx], self_report[train_idx])
        preds_cnn[test_idx] = predict_cnn(net, raw_noisy[test_idx], y_mean, y_std)

    r2_rf = r2_score(self_report, preds_rf)
    r2_cnn = r2_score(self_report, preds_cnn)
    corr_rf = np.corrcoef(self_report, preds_rf)[0, 1]
    corr_cnn = np.corrcoef(self_report, preds_cnn)[0, 1]
    return r2_rf, r2_cnn, corr_rf, corr_cnn


def main():
    print(f"{'Noise sigma':>12}{'RF R^2':>10}{'CNN R^2':>10}{'RF r':>10}{'CNN r':>10}")
    print("-" * 52)
    results = []
    for sigma in [0.0, 0.3, 0.6, 1.0, 1.5]:
        r2_rf, r2_cnn, corr_rf, corr_cnn = evaluate_at_noise_level(sigma)
        results.append((sigma, r2_rf, r2_cnn, corr_rf, corr_cnn))
        print(f"{sigma:>12.1f}{r2_rf:>10.3f}{r2_cnn:>10.3f}{corr_rf:>10.3f}{corr_cnn:>10.3f}")

    rf_drop = results[0][1] - results[-1][1]
    cnn_drop = results[0][2] - results[-1][2]
    print(f"\nR^2 drop from sigma=0.0 to sigma=1.5: RF={rf_drop:.3f}, CNN={cnn_drop:.3f}")
    print(f"{'Classical ML degrades less' if rf_drop < cnn_drop else 'CNN degrades less'} "
          f"under this noise sweep.")


if __name__ == "__main__":
    main()
