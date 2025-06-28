import pandas as pd
import pickle
import os

def get_input_path(year, month):
    default_input_pattern = 'data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = 'data/predictions_yellow_{year:04d}_{month:02d}.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filename):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    if filename.startswith("s3://") and S3_ENDPOINT_URL:
        options = {
            "key": os.getenv("AWS_ACCESS_KEY_ID", "test"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
            "client_kwargs": {
                "endpoint_url": S3_ENDPOINT_URL
            }
        }
        print(f"Reading from S3 with storage options: {options}")
        df = pd.read_parquet(filename, storage_options=options)
    else:
        df = pd.read_parquet(filename)

    return df




def prepare_data(df, categorical):
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].astype(str)

    return df


def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    print(f"📥 Input path: {input_file}")
    print(f"📤 Output path: {output_file}")

    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")  # ✅ Add this line


    with open("model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)

    categorical = ["PULocationID", "DOLocationID"]

    df = read_data(input_file)
    df = prepare_data(df, categorical)

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df_result = pd.DataFrame()
    df_result["ride_id"] = df.index.astype(str)
    df_result["predicted_duration"] = y_pred

    if output_file.startswith("s3://") and S3_ENDPOINT_URL:
        storage_options = {
            "key": os.getenv("AWS_ACCESS_KEY_ID", "test"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
            "client_kwargs": {
                "endpoint_url": S3_ENDPOINT_URL
            }
        }
        print(f"Writing to S3 with storage options: {storage_options}")
        df_result.to_parquet(output_file, index=False, storage_options=storage_options)
    else:
        df_result.to_parquet(output_file, index=False)


if __name__ == "__main__":
    main(2023, 3)
