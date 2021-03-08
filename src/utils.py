import boto3
import pandas as pd


def read_object(s3_path, file_type, **kwargs):
    """Read a file from S3 into a pandas.DataFrame.

    Arguments:
        s3_path (str)

    Returns:
        df (pd.DataFrame)
    """
    # Get S3 client.
    s3 = boto3.client('s3')
    # Get object.
    bucket, key = s3_path.split('/')[2], '/'.join(s3_path.split('/')[3:])
    object = s3.get_object(Bucket=bucket, Key=key)
    # Read object into pandas.DataFrame.
    if file_type == "csv":
        df = pd.read_csv(object['Body'], **kwargs)
    elif file_type == "json":
        df = pd.read_json(object['Body'], lines=True, **kwargs)
    else:
        raise Exception("Only file_type csv and json currently supported.")
    return df


def write_object(df, s3_path, file_type="json"):
    """Save a pandas.DataFrame to S3.

    Arguments:
        df (pd.DataFrame)
        s3_path (str)
    """
    if file_type == "csv":
        df.to_csv(s3_path)
    elif file_type == "json":
        df.to_json(s3_path, lines=True, orient="records")
    else:
        raise Exception("Only file_type csv and json currently supported.")
