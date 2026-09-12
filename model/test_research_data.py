from __future__ import annotations

import unittest

import pandas as pd

from research_data import _parse_dates


class LifecycleDateParsingTests(unittest.TestCase):
    def test_month_year_lendingclub_dates_parse_to_first_of_month(self) -> None:
        parsed = _parse_dates(pd.Series(["Dec-11", "Jan-12", "Jun-07", None]))
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2011-12-01"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2012-01-01"))
        self.assertEqual(parsed.iloc[2], pd.Timestamp("2007-06-01"))
        self.assertTrue(pd.isna(parsed.iloc[3]))


if __name__ == "__main__":
    unittest.main()
