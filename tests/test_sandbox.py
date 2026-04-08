import pandas as pd
from core.sandbox import safe_exec, validate_code

def test_validate_code_safe():
    code = "df_result = df.copy()\ndf_result['new_col'] = 1"
    violations = validate_code(code)
    assert not violations

def test_validate_code_unsafe():
    code = "import os\nos.system('echo malicious')"
    violations = validate_code(code)
    assert len(violations) > 0
    assert "Blocked pattern detected" in violations[0]

def test_safe_exec_success():
    df = pd.DataFrame({"a": [1, 2, 3]})
    code = "df_result = df.copy()\ndf_result['b'] = df_result['a'] * 2"
    result = safe_exec(code, df)
    
    assert result.success is True
    assert "b" in result.result.columns
    assert list(result.result["b"]) == [2, 4, 6]

def test_safe_exec_error():
    df = pd.DataFrame({"a": [1, 2, 3]})
    # Typo: df_result is not defined properly or syntax error
    code = "df_resut = df.copy()\n1/0"
    result = safe_exec(code, df)
    
    assert result.success is False
    assert "ZeroDivisionError" in result.error
