import os
import pandas as pd


def main():
    # Set environment variables for input/output paths and S3 endpoint
    os.environ["INPUT_FILE_PATTERN"] = "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
    os.environ["OUTPUT_FILE_PATTERN"] = "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
    os.environ["S3_ENDPOINT_URL"] = "http://localhost:4566"

    # Run the batch pipeline for Jan 2023 using system call
    exit_code = os.system("python batch.py")
    assert exit_code == 0

    # Read result back from Localstack S3
    output_path = "s3://nyc-duration/out/2023-01.parquet"
    options = {
        'key': 'test',
        'secret': 'test',
        'client_kwargs': {
            'endpoint_url': "http://localhost:4566"
        }
    }

    df_result = pd.read_parquet(output_path, storage_options=options)

    print("\n📄 Full predictions DataFrame:")
    print(df_result)

    print("\n🔢 Individual predictions:")
    print(df_result["predicted_duration"].to_list())

    total_duration = df_result["predicted_duration"].sum()
    print(f"\n🧮 ✅ Total predicted duration: {total_duration:.2f}")


if __name__ == "__main__":
    main()
