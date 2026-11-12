import numpy as np
import pytest

from src.m1_data.nsch import three_class_from_codes
from src.m5_models.multiclass import CLASS_NAMES, MULTICLASS_MODELS
from src.m8_eval import multiclass_metrics

NAN = np.nan


def test_three_class_rule_and_collapse_to_binary():
    #            dep_ever dep_cur anx_ever anx_cur  expected
    cases = [(2, NAN, 2, NAN, 0), (2, NAN, 1, 1, 1), (2, NAN, 1, 2, 0), (1, 1, 2, NAN, 2),
             (1, 1, 1, 1, 2), (1, 2, 1, 1, 1), (1, 1, NAN, NAN, 2),   # depression decides even if anxiety missing
             (2, NAN, NAN, NAN, NAN), (2, NAN, 1, NAN, NAN), (1, NAN, 2, NAN, NAN)]
    cols = list(zip(*cases))
    out = three_class_from_codes(*cols[:4]).to_numpy()
    expected = np.array(cols[4], dtype=float)
    assert np.array_equal(np.isnan(out), np.isnan(expected))
    ok = ~np.isnan(expected)
    assert np.array_equal(out[ok], expected[ok])
    dep_binary = np.array([0, 0, 0, 1, 1, 0, 1], dtype=float)           # depression_current of the determined rows
    assert np.array_equal((out[ok] == 2).astype(float), dep_binary)


def test_multiclass_metrics_known_values():
    y = np.array([0, 0, 1, 1, 2, 2])
    perfect = np.eye(3)[y]
    m = multiclass_metrics(y, perfect, y, CLASS_NAMES)
    assert m["macro_f1"] == 1 and m["macro_ovr_auc"] == 1 and m["balanced_accuracy"] == 1
    assert "accuracy" not in m
    m = multiclass_metrics(y, perfect, np.zeros(6, int), CLASS_NAMES)
    assert m["balanced_accuracy"] == pytest.approx(1 / 3) and m["recall_none"] == 1


@pytest.mark.parametrize("name", sorted(MULTICLASS_MODELS))
def test_multiclass_models_learn_three_imbalanced_classes(name):
    rng = np.random.default_rng(0)
    y = rng.choice(3, size=300, p=[0.78, 0.12, 0.10])
    X = rng.normal(size=(300, 4)) * 0.3 + y[:, None] * 0.9    # class signal on every feature
    X = np.clip(X - X.min(0), 0, None) / (X.max(0) - X.min(0)) * np.pi    # angle-scaled, as for the QSVM
    params = {"n_qubits": 4} if name in ("qsvm", "vqc") else {}
    if name == "vqc":
        params.update(epochs=30, batch_size=16, learning_rate=0.1)
    model = MULTICLASS_MODELS[name](seed=0, **params)
    if getattr(model, "supports_eval_set", False):
        model.fit(X[:160], y[:160], eval_set=(X[160:200], y[160:200]))
    else:
        model.fit(X[:200], y[:200])
    scores = model.class_scores(X[200:])
    assert scores.shape == (100, 3)
    m = multiclass_metrics(y[200:], scores, model.predict(X[200:]), CLASS_NAMES)
    assert m["macro_ovr_auc"] > 0.7


def test_holdout_rule_three_class_excludes_undetermined_binary_uses_all_rows():
    from src.m10_exp.exp9_multiclass import HOLDOUT_POLICY, holdout_scoring_rows

    y3 = np.array([0, 1, 2, NAN, 0, NAN, 2, 1], dtype=float)
    holdout = np.array([1, 3, 4, 5, 6])
    assert HOLDOUT_POLICY == {"three_class_metrics": "exclude_undetermined", "binary_depression_metrics": "all_rows"}
    assert holdout_scoring_rows(holdout, y3, "three_class_metrics").tolist() == [1, 4, 6]
    assert holdout_scoring_rows(holdout, y3, "binary_depression_metrics").tolist() == [1, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        holdout_scoring_rows(holdout, y3, "accuracy")


@pytest.mark.parametrize("path", ["configs/exp9_multiclass_nsch2024.yaml", "configs/exp9b_multiclass_vqc_nsch2024.yaml"])
def test_exp9_configs_carry_the_prespecified_holdout_rule_and_others_are_refused(path):
    import copy

    import yaml

    from src.m10_exp.exp9_multiclass import check_holdout_policy

    config = yaml.safe_load(open(path, encoding="utf-8"))
    check_holdout_policy(config)                                   # passes as committed
    for bad in ({"three_class_metrics": "all_rows", "binary_depression_metrics": "all_rows"}, None):
        altered = copy.deepcopy(config)
        if bad is None:
            altered.pop("holdout_policy")
        else:
            altered["holdout_policy"] = bad
        with pytest.raises(ValueError):
            check_holdout_policy(altered)


def test_undetermined_rows_have_a_defined_binary_label():
    # The rule relies on this: whenever the three-class label is undetermined, the binary depression label
    # (same rule as the binary loader) is determined and negative. Checked over every code combination.
    import itertools

    import pandas as pd

    from src.m1_data.nsch import _current

    codes = [1, 2, NAN, 6]                                         # yes, no, missing, an invalid code
    combos = list(itertools.product(codes, repeat=4))
    de, dc, ae, ac = (list(c) for c in zip(*combos))
    y3 = three_class_from_codes(de, dc, ae, ac).to_numpy()
    dep = _current(pd.Series(de, dtype=float), pd.Series(dc, dtype=float)).to_numpy()
    undetermined_with_dep_label = np.isnan(y3) & ~np.isnan(dep)
    assert undetermined_with_dep_label.any()
    assert (dep[undetermined_with_dep_label] == 0).all()
