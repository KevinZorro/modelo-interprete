from lsc70 import t07_optuna as t7


def test_objective_penalises_guard_rail_violations():
    cfg = {"f1_floor": 0.86, "fr_toma_ceiling": 0.10}
    ok = {"fa": 0.06, "f1_macro": 0.87, "fr_toma": 0.05}
    assert t7.objective_value(ok, cfg) == 0.06
    assert t7.objective_value({**ok, "f1_macro": 0.80}, cfg) > 0.6
    assert t7.objective_value({**ok, "fr_toma": 0.20}, cfg) > 1.0


def test_control_matches_notebook_network():
    from lsc70.entrenar_evaluar import LEARNING_RATE

    assert (t7.CONTROL["units1"], t7.CONTROL["units2"]) == (256, 128)
    assert (t7.CONTROL["dropout1"], t7.CONTROL["dropout2"]) == (0.4, 0.3)
    assert t7.CONTROL["lr"] == LEARNING_RATE
