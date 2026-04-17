"""Tests for apex.data.fred_client (mocked, no real API calls)."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from apex.data.fred_client import fetch_fred_series, _get_fred_api_key


class TestGetFredApiKey:
    def test_env_var_takes_priority(self):
        with patch.dict("os.environ", {"FRED_API_KEY": "env_key"}):
            assert _get_fred_api_key() == "env_key"

    def test_falls_back_to_cfg(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("apex.data.fred_client.CFG", {"fred_api_key": "cfg_key"}):
                assert _get_fred_api_key() == "cfg_key"

    def test_raises_when_no_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("apex.data.fred_client.CFG", {}):
                with pytest.raises(RuntimeError, match="FRED API key not found"):
                    _get_fred_api_key()


class TestFetchFredSeries:
    def _mock_response(self, observations):
        """Build a mock requests.get response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"observations": observations}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    @patch("apex.data.fred_client.time.sleep")
    @patch("apex.data.fred_client.requests.get")
    @patch("apex.data.fred_client._get_fred_api_key", return_value="test_key")
    def test_url_construction(self, mock_key, mock_get, mock_sleep, tmp_path):
        """Verify the correct URL and params are passed to requests.get."""
        mock_get.return_value = self._mock_response([
            {"date": "2024-01-02", "value": "18.5"},
        ])

        df = fetch_fred_series("VIXCLS", "2024-01-01", "2024-01-31",
                               cache_dir=tmp_path)

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "api.stlouisfed.org" in call_args[0][0] or \
               "api.stlouisfed.org" in str(call_args)
        params = call_args[1].get("params", call_args[0][1] if len(call_args[0]) > 1 else {})
        if not params:
            params = call_args[1] if "series_id" in call_args[1] else call_args[1].get("params", {})
        # Verify data returned
        assert len(df) == 1
        assert df["value"].iloc[0] == 18.5

    @patch("apex.data.fred_client.time.sleep")
    @patch("apex.data.fred_client.requests.get")
    @patch("apex.data.fred_client._get_fred_api_key", return_value="test_key")
    def test_missing_values_filtered(self, mock_key, mock_get, mock_sleep, tmp_path):
        """Missing FRED values (\".\") should be dropped."""
        mock_get.return_value = self._mock_response([
            {"date": "2024-01-02", "value": "18.5"},
            {"date": "2024-01-03", "value": "."},
            {"date": "2024-01-04", "value": "19.2"},
            {"date": "2024-01-05", "value": ""},
        ])

        df = fetch_fred_series("VIXCLS", "2024-01-01", "2024-01-31",
                               cache_dir=tmp_path)
        assert len(df) == 2
        assert list(df["value"]) == [18.5, 19.2]

    @patch("apex.data.fred_client.time.sleep")
    @patch("apex.data.fred_client.requests.get")
    @patch("apex.data.fred_client._get_fred_api_key", return_value="test_key")
    def test_cache_hit(self, mock_key, mock_get, mock_sleep, tmp_path):
        """Second call should use cache and not hit the API."""
        mock_get.return_value = self._mock_response([
            {"date": "2024-01-02", "value": "18.5"},
        ])

        # First call fetches
        df1 = fetch_fred_series("VIXCLS", "2024-01-01", "2024-01-31",
                                cache_dir=tmp_path)
        assert mock_get.call_count == 1

        # Second call should use cache
        df2 = fetch_fred_series("VIXCLS", "2024-01-01", "2024-01-31",
                                cache_dir=tmp_path)
        assert mock_get.call_count == 1  # no additional call
        assert len(df2) == 1

    @patch("apex.data.fred_client.time.sleep")
    @patch("apex.data.fred_client.requests.get")
    @patch("apex.data.fred_client._get_fred_api_key", return_value="test_key")
    def test_empty_response(self, mock_key, mock_get, mock_sleep, tmp_path):
        """Empty observations should return empty DataFrame."""
        mock_get.return_value = self._mock_response([])
        df = fetch_fred_series("VIXCLS", "2024-01-01", "2024-01-31",
                               cache_dir=tmp_path)
        assert len(df) == 0
