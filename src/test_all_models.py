"""
Run all model tests sequentially and print a comparative summary table.

Usage:
    python src/test_all_models.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import importlib

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_TESTS = [
    ('Logistic Regression', 'test_logistic_regression'),
    ('Random Forest',       'test_random_forest'),
    ('XGBoost',             'test_xgboost'),
    ('BiLSTM',              'test_bilstm'),
    ('GRU',                 'test_gru'),
]


def main():
    results = {}

    print("=" * 60)
    print("  RUNNING ALL MODEL TESTS — 2023 Holdout")
    print("=" * 60)

    for display_name, module_name in MODEL_TESTS:
        print(f"\n▶ Testing {display_name}...")
        try:
            mod = importlib.import_module(module_name)
            metrics = mod.test()
            if metrics is not None:
                results[display_name] = metrics
            else:
                results[display_name] = None
                print(f"  ⚠ Skipped (model not found)")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[display_name] = None

    # ── Summary Table ──
    print(f"\n\n{'=' * 60}")
    print("  COMPARATIVE RESULTS — 2023 Holdout")
    print(f"{'=' * 60}")

    # Header
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─' * 65}")

    for name, metrics in results.items():
        if metrics is None:
            print(f"  {name:<25} {'—':>10} {'—':>10} {'—':>10} {'—':>10}")
        else:
            print(f"  {name:<25} {metrics['Accuracy']:>10.4f} {metrics['Precision']:>10.4f} {metrics['Recall']:>10.4f} {metrics['F1']:>10.4f}")

    print(f"  {'─' * 65}")

    # Best model
    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        best = max(valid, key=lambda k: valid[k]['F1'])
        print(f"\n  🏆 Best Model (by F1): {best} — F1 = {valid[best]['F1']:.4f}")

    print()


if __name__ == '__main__':
    main()
