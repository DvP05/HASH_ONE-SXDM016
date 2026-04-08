from tools.api_tools import fetch_nasa_firms


def test_fetch_nasa_firms_fallback_without_key(monkeypatch):
    monkeypatch.delenv("NASA_API_KEY", raising=False)

    df = fetch_nasa_firms()

    assert not df.empty
    assert "confidence" in df.columns
    assert "data_source" in df.columns
    assert (df["data_source"] == "firms_fallback").all()
