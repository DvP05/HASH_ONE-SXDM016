"""
Scheduler — Hourly NASA API data refresh and prediction update loop.

Runs NASA FIRMS, EONET, and POWER API calls every hour (configurable)
to keep the dashboard and predictions up-to-date with live data.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


class DataRefreshScheduler:
    """
    Background scheduler that periodically:
    1. Fetches fresh data from NASA APIs (FIRMS, EONET, POWER)
    2. Saves timestamped cache files
    3. Optionally re-runs the prediction pipeline
    """

    def __init__(self, interval_seconds: int = 3600, cache_dir: str = "data/live_cache",
                 output_dir: str = "output", run_pipeline_callback=None):
        self.interval = interval_seconds
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.run_pipeline_callback = run_pipeline_callback
        self._timer: threading.Timer | None = None
        self._running = False
        self._refresh_count = 0
        self.last_refresh: str = ""
        self.next_refresh: str = ""
        self._lock = threading.Lock()

        os.makedirs(cache_dir, exist_ok=True)

    def start(self):
        """Start the periodic refresh scheduler."""
        self._running = True
        logger.info(f"🔄 Scheduler started — refreshing every {self.interval}s ({self.interval / 3600:.1f}h)")
        self._schedule_next()

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
        logger.info("⏹️ Scheduler stopped")

    def _schedule_next(self):
        """Schedule the next refresh cycle."""
        if not self._running:
            return
        self._timer = threading.Timer(self.interval, self._execute_refresh)
        self._timer.daemon = True
        self._timer.start()
        self.next_refresh = datetime.now(timezone.utc).isoformat()

    def _execute_refresh(self):
        """Execute one refresh cycle."""
        with self._lock:
            self._refresh_count += 1
            cycle = self._refresh_count
            ts = datetime.now(timezone.utc)
            self.last_refresh = ts.isoformat()

            logger.info(f"\n{'━' * 50}")
            logger.info(f"🔄 DATA REFRESH CYCLE #{cycle} @ {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            logger.info(f"{'━' * 50}")

            try:
                self._refresh_nasa_data(ts)
            except Exception as e:
                logger.error(f"❌ Refresh cycle #{cycle} failed: {e}")

            # Re-run pipeline if callback provided
            if self.run_pipeline_callback:
                try:
                    logger.info("🔄 Re-running prediction pipeline...")
                    self.run_pipeline_callback()
                    logger.info("✅ Pipeline re-run complete")
                except Exception as e:
                    logger.error(f"❌ Pipeline re-run failed: {e}")

            logger.info(f"✅ Refresh cycle #{cycle} complete")

        # Schedule next
        self._schedule_next()

    def _refresh_nasa_data(self, timestamp: datetime):
        """Fetch fresh data from all NASA APIs and cache it."""
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")

        # Import here to avoid circular imports
        from tools.api_tools import (
            fetch_nasa_firms_global,
            fetch_nasa_eonet,
            fetch_nasa_power_global,
        )

        # ── FIRMS Global ──
        try:
            logger.info("🛰️ Fetching NASA FIRMS (global)...")
            firms_df = fetch_nasa_firms_global()
            if firms_df is not None and not firms_df.empty:
                path = os.path.join(self.cache_dir, f"firms_{ts_str}.csv")
                firms_df.to_csv(path, index=False)
                # Also update the "latest" symlink
                latest = os.path.join(self.cache_dir, "firms_latest.csv")
                firms_df.to_csv(latest, index=False)
                logger.info(f"   FIRMS: {len(firms_df)} records saved")
        except Exception as e:
            logger.warning(f"   FIRMS fetch failed: {e}")

        # ── EONET Events ──
        try:
            logger.info("🌍 Fetching NASA EONET events...")
            eonet_df = fetch_nasa_eonet(days=30, limit=100)
            if eonet_df is not None and not eonet_df.empty:
                path = os.path.join(self.cache_dir, f"eonet_{ts_str}.csv")
                eonet_df.to_csv(path, index=False)
                latest = os.path.join(self.cache_dir, "eonet_latest.csv")
                eonet_df.to_csv(latest, index=False)
                logger.info(f"   EONET: {len(eonet_df)} events saved")
        except Exception as e:
            logger.warning(f"   EONET fetch failed: {e}")

        # ── POWER Global ──
        try:
            logger.info("🌤️ Fetching NASA POWER (global grid)...")
            power_df = fetch_nasa_power_global()
            if power_df is not None and not power_df.empty:
                path = os.path.join(self.cache_dir, f"power_{ts_str}.csv")
                power_df.to_csv(path, index=False)
                latest = os.path.join(self.cache_dir, "power_latest.csv")
                power_df.to_csv(latest, index=False)
                logger.info(f"   POWER: {len(power_df)} records saved")
        except Exception as e:
            logger.warning(f"   POWER fetch failed: {e}")

        # Cleanup old cache files (keep last 24)
        self._cleanup_cache(keep=24)

    def _cleanup_cache(self, keep: int = 24):
        """Remove oldest cache files, keeping the N most recent per source."""
        for prefix in ("firms_2", "eonet_2", "power_2"):  # 2 matches year prefix e.g. 20260408
            files = sorted([
                f for f in os.listdir(self.cache_dir)
                if f.startswith(prefix) and f.endswith(".csv")
            ])
            if len(files) > keep:
                for old_file in files[:-keep]:
                    try:
                        os.remove(os.path.join(self.cache_dir, old_file))
                    except OSError:
                        pass

    def get_status(self) -> dict:
        """Get the current scheduler status."""
        return {
            "running": self._running,
            "interval_seconds": self.interval,
            "refresh_count": self._refresh_count,
            "last_refresh": self.last_refresh,
            "next_refresh": self.next_refresh,
        }

    def run_now(self):
        """Trigger an immediate refresh (in addition to the scheduled ones)."""
        threading.Thread(target=self._execute_refresh, daemon=True).start()


def load_latest_cache(cache_dir: str = "data/live_cache") -> dict:
    """
    Load the latest cached data files.
    Returns a dict with 'firms', 'eonet', 'power' DataFrames.
    """
    result = {}

    for source in ("firms", "eonet", "power"):
        latest_path = os.path.join(cache_dir, f"{source}_latest.csv")
        if os.path.exists(latest_path):
            try:
                result[source] = pd.read_csv(latest_path)
                logger.info(f"Loaded cached {source}: {len(result[source])} rows")
            except Exception as e:
                logger.warning(f"Failed to load {source} cache: {e}")

    return result
