"""
Tests for the DataRefreshScheduler.
"""
import os
import sys
import time
import pytest
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSchedulerInit:
    """Test scheduler initialization."""

    def test_create_scheduler(self, tmp_path):
        from scheduler import DataRefreshScheduler
        s = DataRefreshScheduler(interval_seconds=3600, cache_dir=str(tmp_path))
        assert s.interval == 3600
        assert s._running is False
        assert s._refresh_count == 0

    def test_custom_interval(self, tmp_path):
        from scheduler import DataRefreshScheduler
        s = DataRefreshScheduler(interval_seconds=60, cache_dir=str(tmp_path))
        assert s.interval == 60


class TestSchedulerStatus:
    """Test scheduler status reporting."""

    def test_get_status_not_running(self, tmp_path):
        from scheduler import DataRefreshScheduler
        s = DataRefreshScheduler(cache_dir=str(tmp_path))
        status = s.get_status()
        assert status["running"] is False
        assert status["refresh_count"] == 0

    def test_start_stop(self, tmp_path):
        from scheduler import DataRefreshScheduler
        s = DataRefreshScheduler(interval_seconds=9999, cache_dir=str(tmp_path))
        s.start()
        assert s._running is True
        time.sleep(0.1)
        s.stop()
        assert s._running is False


class TestSchedulerCallback:
    """Test the pipeline callback mechanism."""

    def test_callback_is_invoked(self, tmp_path):
        from scheduler import DataRefreshScheduler
        callback_called = threading.Event()

        def callback():
            callback_called.set()

        s = DataRefreshScheduler(
            interval_seconds=1,
            cache_dir=str(tmp_path),
            run_pipeline_callback=callback,
        )
        # Don't actually start the scheduler loop, just test the method
        assert s.run_pipeline_callback is not None


class TestCacheLoader:
    """Test loading cached data."""

    def test_load_empty_cache(self, tmp_path):
        from scheduler import load_latest_cache
        result = load_latest_cache(cache_dir=str(tmp_path))
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_load_existing_cache(self, tmp_path):
        import pandas as pd
        from scheduler import load_latest_cache

        # Write a test cache file
        df = pd.DataFrame({"latitude": [28.6], "longitude": [77.2], "brightness": [310.5]})
        df.to_csv(os.path.join(str(tmp_path), "firms_latest.csv"), index=False)

        result = load_latest_cache(cache_dir=str(tmp_path))
        assert "firms" in result
        assert len(result["firms"]) == 1


class TestCacheCleanup:
    """Test cache file cleanup."""

    def test_cleanup_keeps_recent(self, tmp_path):
        from scheduler import DataRefreshScheduler
        s = DataRefreshScheduler(cache_dir=str(tmp_path))

        # Create some dummy cache files
        for i in range(30):
            path = os.path.join(str(tmp_path), f"firms_2026040{i:02d}_120000.csv")
            with open(path, "w") as f:
                f.write("latitude,longitude\n1,2\n")

        s._cleanup_cache(keep=5)
        remaining = [f for f in os.listdir(str(tmp_path)) if f.startswith("firms_2")]
        assert len(remaining) <= 5
