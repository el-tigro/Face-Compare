from fastapi import HTTPException
import mysql.connector
import os
from dotenv import load_dotenv
import boto3
import datetime as dt
import time
import logging
import numpy as np
import pandas as pd
from functools import wraps
import warnings
from PIL.TiffImagePlugin import IFDRational

warnings.filterwarnings("ignore")


# S3 Keys
bucket_name_dict = {
    "company_name1": "prod-company_name1.ru",
    "company_name2": "prod-company_name2.ru"
}
logger = logging.getLogger(__name__)
load_dotenv()

endpoint_url = "https://s3.ru-1.bucket.companycloud.ru"
aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
region_name = 'ru-1'
prefix = "uploads/"
download_folder = "photo_examples/"  # Folder where photos will be saved


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Execution time of {func.__name__}: {execution_time:.4f} seconds")
        return result

    return wrapper


def convert_data(data):
    """
    Data type conversion for working with FastAPI
    """
    if isinstance(data, dict):
        return {key: convert_data(value) for key, value in data.items()}
    elif isinstance(data, tuple):
        return tuple(convert_data(value) for value in data)
    elif isinstance(data, list):
        return [convert_data(value) for value in data]
    elif isinstance(data, IFDRational):
        if hasattr(data, "denominator") and data.denominator != 0:
            return float(data)
        else:
            return None
    elif type(data) in [np.float16, np.float32, np.float64]:
        return float(data)
    elif type(data) in [np.int16, np.int32, np.int64, np.uint8]:
        return int(data)
    else:
        return data


@time_it
def mysql_query(company: str, sql: str) -> pd.DataFrame:
    if company.lower() in ["COMPANY_NAME1".lower(), "CN1".lower()]:
        host_id = "0.0.0.17"
        database_id = "prod_company_name1_api"
    if company.lower() in ["COMPANY_NAME2".lower(), "CN2".lower()]:
        host_id = "0.0.0.37"
        database_id = "prod_company_name2_api"
    conn = mysql.connector.connect(
        host=host_id,
        database=database_id,
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


@time_it
def load_hashedFilename(company_name, loan_id):
    sql_text = f"""
    select 
        loanid as loan_id,
        file.id as file_id, file.userId as user_id, 
        file.name, file.insertDate, file.hashedFilename, file.validation_status
    from (
        select 
            id as loanid, userId
        from loan as l
        where l.id = {loan_id}
    ) as l 
    left join file on file.userId=l.userId and `type` = 'user'
    where hashedFilename is not null
    """
    try:
        df = mysql_query(company_name, sql_text)
        logger.info(f"Downloaded dataframe, shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error while loading file from S3: {e}")
        raise HTTPException(status_code=500, detail="Database connection issue")


@time_it
def download_file_from_s3(
        company_name,
        hashed_file_name,
        download_folder
):
    bucket_name = bucket_name_dict[company_name]
    # Create a session and S3 client with a custom endpoint
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name
    )
    s3 = session.client('s3', endpoint_url=endpoint_url)

    try:
        # Download the file from S3
        download_path = download_folder + hashed_file_name
        object_key = prefix + hashed_file_name
        s3.download_file(bucket_name, object_key, download_path)
        logger.info(f"File {object_key} successfully downloaded to {download_path}")
        return download_path

    except Exception as e:
        logger.error(f"Error while downloading file from S3: {e}")
        raise HTTPException(status_code=500, detail="S3 connection issue")


@time_it
def remove_file_from_s3(downloaded_image_path):
    if os.path.exists(downloaded_image_path):
        os.remove(downloaded_image_path)
        logger.info(f"File successfully deleted from {downloaded_image_path}")
    return None