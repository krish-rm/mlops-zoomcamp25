import pandas as pd
import pickle
import os


def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[categorical] = df[categorical].astype(str)

    return df


def main(year, month):
    input_file = f"data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"data/predictions_yellow_{year:04d}_{month:02d}.parquet"

    with open("model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)

    categorical = ["PULocationID", "DOLocationID"]
    df = read_data(input_file, categorical)

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df_result = pd.DataFrame()
    df_result["ride_id"] = df.index.astype(str)
    df_result["predicted_duration"] = y_pred

    df_result.to_parquet(output_file, index=False)
    print(f"✅ Predictions saved to {output_file}")


if __name__ == "__main__":
    main(2023, 3)
