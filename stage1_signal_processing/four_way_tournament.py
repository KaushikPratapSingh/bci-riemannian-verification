import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from phase4_tournament import simulate_session, extract_features  # noqa: E402
from riemannian import (covariance_features, riemannian_mean,  # noqa: E402
                         invsqrtm_spd, tangent_space_vector, sanity_check)


def run_four_way_tournament(n_sessions=50, seed=7):
    assert sanity_check(), "Riemannian implementation failed its sanity check -- not trusting it."

    rng = np.random.default_rng(seed)
    true_focus = rng.uniform(0, 1, n_sessions)
    raw_windows = np.array([simulate_session(f, seed=i) for i, f in enumerate(true_focus)])
    X_feat = np.array([extract_features(w) for w in raw_windows])
    covs = np.array([covariance_features(w) for w in raw_windows])
    self_report = np.clip(np.round(1 + true_focus * 4 + rng.normal(0, 0.4, n_sessions)), 1, 5)
    shuffled_report = rng.permutation(self_report)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    results = {}
    for label_name, labels in [("intact", self_report), ("shuffled", shuffled_report)]:
        preds_rf = np.zeros(n_sessions)
        preds_riemann = np.zeros(n_sessions)
        for tr, te in kf.split(X_feat):
            rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=0)
            rf.fit(X_feat[tr], labels[tr])
            preds_rf[te] = rf.predict(X_feat[te])

            # Riemannian tangent-space Ridge: fit the geometric mean on
            # TRAINING covariances only (no test-set leakage into the
            # reference point), project both train and test onto that
            # tangent space, then run Ridge on the resulting vectors.
            C_mean = riemannian_mean(covs[tr])
            C_mean_invsqrt = invsqrtm_spd(C_mean)
            tangent_tr = np.array([tangent_space_vector(c, C_mean_invsqrt) for c in covs[tr]])
            tangent_te = np.array([tangent_space_vector(c, C_mean_invsqrt) for c in covs[te]])
            ridge_riemann = Ridge(alpha=1.0)
            ridge_riemann.fit(tangent_tr, labels[tr])
            preds_riemann[te] = ridge_riemann.predict(tangent_te)

        results[label_name] = {
            "rf_r2": r2_score(labels, preds_rf),
            "rf_r": np.corrcoef(labels, preds_rf)[0, 1],
            "riemann_r2": r2_score(labels, preds_riemann),
            "riemann_r": np.corrcoef(labels, preds_riemann)[0, 1],
        }

    print(f"{'Labels':<12}{'Model':<28}{'R^2':>8}{'r':>10}")
    print("-" * 58)
    for label_name, r in results.items():
        print(f"{label_name:<12}{'Random Forest':<28}{r['rf_r2']:>8.3f}{r['rf_r']:>10.3f}")
        print(f"{label_name:<12}{'Riemannian tangent+Ridge':<28}{r['riemann_r2']:>8.3f}{r['riemann_r']:>10.3f}")

    print("\nNegative-control gate:")
    for model in ["rf", "riemann"]:
        r2 = results["shuffled"][f"{model}_r2"]
        print(f"  {model}: shuffled R^2={r2:.3f} -> {'PASS' if r2 < 0 else 'FAIL'}")


if __name__ == "__main__":
    run_four_way_tournament()
