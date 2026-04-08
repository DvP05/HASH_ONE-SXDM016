import pytest
from orchestrator.quality_gates import QualityGates

class DummyProfile:
    def __init__(self, missing_pct):
        self.missing_pct = missing_pct

class DummyResult:
    def __init__(self, missing_pct=None, roc_auc=None, r2=None, task_type="classification"):
        if missing_pct is not None:
            self.profile_after = DummyProfile(missing_pct)
        else:
            self.profile_after = None
            
        if roc_auc is not None or r2 is not None:
            class Eval:
                def __init__(self, a, r, tt):
                    self.roc_auc = a
                    self.r2 = r
                    self.task_type = tt
            self.evaluation = Eval(roc_auc, r2, task_type)
        else:
            self.evaluation = None

def test_gate_data_cleaning():
    gates = QualityGates()
    
    # Should pass (0.01 missing < 0.05 default)
    res_pass = DummyResult(missing_pct={"col1": 0.01})
    gate_pass = gates.validate("data_cleaning", res_pass)
    assert gate_pass.passed is True
    
    # Should fail (0.10 missing > 0.05 default)
    res_fail = DummyResult(missing_pct={"col1": 0.10})
    gate_fail = gates.validate("data_cleaning", res_fail)
    assert gate_fail.passed is False
    assert len(gate_fail.issues) == 1

def test_gate_modeling_classification():
    gates = QualityGates()
    
    # Should pass (0.8 > 0.55 default)
    res_pass = DummyResult(roc_auc=0.8)
    gate_pass = gates.validate("modeling", res_pass)
    assert gate_pass.passed is True
    
    # Should fail (0.5 < 0.55 default)
    res_fail = DummyResult(roc_auc=0.5)
    gate_fail = gates.validate("modeling", res_fail)
    assert gate_fail.passed is False
