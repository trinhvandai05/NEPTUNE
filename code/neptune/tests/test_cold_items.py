import numpy as np
import pandas as pd

from neptune.data.ml25m import load_positive_events
from neptune.pipeline.prepare import cold_items_from_first_seen


def test_cold_status_uses_the_full_corpus_and_any_rating(tmp_path):
    """Item 20 is rated before T_cut only by a user who will not be in the cohort,
    and only with a LOW rating.  It must still count as warm."""
    pd.DataFrame({"userId": [1, 1, 2, 3], "movieId": [10, 20, 20, 30],
                  "rating": [5.0, 2.0, 5.0, 5.0], "timestamp": [100, 100, 900, 950]}
                 ).to_csv(tmp_path / "ratings.csv", index=False)
    ev, first = load_positive_events(tmp_path, np.array([10, 20, 30, 40]), 4.0, return_first_seen=True)
    assert set(ev["item"]) == {0, 1, 2}                   # the 2.0 rating is not a positive ...
    cold = cold_items_from_first_seen(first, t_cut=500)
    assert cold.tolist() == [False, False, True, True]    # ... but it still makes item 20 warm
