import importlib

# Ensure modules import without heavy side-effects

def test_import_helpers():
    ood_utils = importlib.import_module("spiramycel.ood_utils")
    ood_analysis = importlib.import_module("spiramycel.ood_analysis")

    # Public API attributes
    assert hasattr(ood_utils, "load_trained_models")
    assert hasattr(ood_utils, "load_ood_test_set")
    assert hasattr(ood_analysis, "perform_statistical_analysis")


def test_effect_size_calculation():
    from spiramycel.ood_analysis import calc_effect_size

    g1 = [0.1, 0.2, 0.3]
    g2 = [0.4, 0.5, 0.6]
    d = calc_effect_size(g1, g2)
    # Simple sanity: positive effect size and non-zero
    assert d > 0.0 