from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from weather_extremes.data_loader import parse_dly, values_to_wide
from weather_extremes.features import add_rolling_features
from weather_extremes.labeling import label_extremes
from weather_extremes.model import time_split


def make_dly_line(station_id, year, month, element, values):
    blocks = []
    for value in values:
        if value is None:
            blocks.append(f"{-9999:>5}   ")
        else:
            blocks.append(f"{value:>5}   ")
    blocks.extend([" " * 8] * (31 - len(values)))
    return f"{station_id:<11}{year:>4}{month:>2}{element:<4}" + "".join(blocks)


class ParseDlyTests(unittest.TestCase):
    def test_parse_round_trip(self):
        station_id = "USW00094728"
        line = make_dly_line(station_id, 2020, 1, "TMAX", [100, 200, None, 300])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sample.dly")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(line + "\n")
            frame = parse_dly(path, station_id=station_id)
        self.assertEqual(frame.shape[0], 3)
        self.assertEqual(frame.iloc[0]["date"], pd.Timestamp("2020-01-01"))
        self.assertEqual(frame.iloc[0]["value"], 100)
        self.assertEqual(frame.iloc[2]["value"], 300)

    def test_values_to_wide_metric(self):
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
                "element": ["TMAX", "TMAX"],
                "value": [100, 200],
                "mflag": [None, None],
                "qflag": [None, None],
                "sflag": ["0", "0"],
            }
        )
        wide = values_to_wide(frame, ["TMAX"], units="metric")
        self.assertAlmostEqual(float(wide.loc[pd.Timestamp("2020-01-01"), "TMAX"]), 10.0)
        self.assertAlmostEqual(float(wide.loc[pd.Timestamp("2020-01-02"), "TMAX"]), 20.0)

    def test_qflag_rows_dropped(self):
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
                "element": ["TMAX", "TMAX"],
                "value": [100, 200],
                "mflag": [None, None],
                "qflag": ["G", None],
                "sflag": ["0", "0"],
            }
        )
        wide = values_to_wide(frame, ["TMAX"], units="metric")
        self.assertNotIn(pd.Timestamp("2020-01-01"), wide.index)
        self.assertAlmostEqual(float(wide.loc[pd.Timestamp("2020-01-02"), "TMAX"]), 20.0)


class LabelingTests(unittest.TestCase):
    def test_label_counts_and_thresholds(self):
        rng = np.random.default_rng(0)
        anomaly = pd.Series(rng.normal(0, 1, size=1000))
        frame = pd.DataFrame({"anomaly": anomaly})
        frame.index = pd.date_range("2020-01-01", periods=1000, freq="D")
        labeled, thresholds = label_extremes(frame, hot_quantile=0.95, cold_quantile=0.05)
        self.assertEqual(thresholds["class_counts"]["normal"], 900)
        self.assertEqual(thresholds["class_counts"]["extreme_hot"], 50)
        self.assertEqual(thresholds["class_counts"]["extreme_cold"], 50)
        self.assertTrue((labeled["target"] == 1).sum() == 50)
        self.assertTrue((labeled["target"] == 2).sum() == 50)


class FeatureTests(unittest.TestCase):
    def test_rolling_is_causal(self):
        index = pd.date_range("2020-01-01", periods=10, freq="D")
        frame = pd.DataFrame({"TMAX": np.arange(10, dtype="float64"), "TMIN": np.arange(10, dtype="float64")}, index=index)
        frame = add_rolling_features(frame, [3], ["TMAX", "TMIN"])
        self.assertAlmostEqual(frame["TMAX_roll3_mean"].iloc[3], (0 + 1 + 2) / 3)
        self.assertAlmostEqual(frame["TMAX_roll3_mean"].iloc[4], (1 + 2 + 3) / 3)
        self.assertTrue(np.isnan(frame["TMAX_roll3_mean"].iloc[2]))

    def test_lag_is_causal(self):
        index = pd.date_range("2020-01-01", periods=5, freq="D")
        frame = pd.DataFrame({"TMAX": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
        frame["TMAX_lag1"] = frame["TMAX"].shift(1)
        self.assertTrue(np.isnan(frame["TMAX_lag1"].iloc[0]))
        self.assertEqual(frame["TMAX_lag1"].iloc[3], 3.0)


class SplitTests(unittest.TestCase):
    def test_time_split_is_ordered_and_disjoint(self):
        index = pd.date_range("2020-01-01", periods=100, freq="D")
        frame = pd.DataFrame({"value": np.arange(100)}, index=index)
        train, val, test = time_split(frame, 0.6, 0.2, 0.2)
        self.assertEqual(len(train), 60)
        self.assertEqual(len(val), 20)
        self.assertEqual(len(test), 20)
        self.assertLess(train.index.max(), val.index.min())
        self.assertLess(val.index.max(), test.index.min())


if __name__ == "__main__":
    unittest.main()
