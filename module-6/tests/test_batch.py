import sys
import os
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from batch import prepare_data
import pandas as pd



def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),           # 9 min → valid
        (1, 1, dt(1, 2), dt(1, 10)),                 # 8 min → valid
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),        # 0.983 min → invalid (filtered out)
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),            # 1441 min → invalid (filtered out)
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    actual_df = prepare_data(df, categorical)

    # Expected: rows 0 and 1 only
    assert len(actual_df) == 2

    # Check type casting
    assert actual_df['PULocationID'].dtype == 'object'
    assert actual_df['DOLocationID'].dtype == 'object'

    # Check durations
    durations = actual_df['duration'].tolist()
    assert all(1 <= d <= 60 for d in durations)
