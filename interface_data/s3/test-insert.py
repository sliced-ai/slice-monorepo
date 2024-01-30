import os
import uuid
import tarfile
import mysql.connector
import boto3

def upload_to_s3(bucket, s3_key, local_path):
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_path, bucket, s3_key)

def delete_from_s3(bucket, s3_key):
    s3_client = boto3.client('s3')
    s3_client.delete_object(Bucket=bucket, Key=s3_key)

def main(model_folder, s3_bucket, db_host, db_port, db_user, db_password):
    # Generate UUID
    model_uuid = str(uuid.uuid4())

    # Connect to the database
    connection = mysql.connector.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database="slice",
    )
    cursor = connection.cursor()

    # Insert into model_meta table
    folder_name = os.path.basename(model_folder)
    cursor.execute("INSERT INTO model_meta (uuid, name) VALUES (%s, %s)", (model_uuid, folder_name))
    connection.commit()

    # Create tar file
    tar_path = os.path.join(model_folder, model_uuid)
    with tarfile.open(tar_path, 'w') as tarf:
        for root, _, files in os.walk(model_folder):
            for file in files:
                if not file.endswith('.pth'):
                    file_path = os.path.join(root, file)
                    tarf.add(file_path, arcname=os.path.basename(file_path))

    # Upload to S3
    upload_to_s3(s3_bucket, model_uuid, tar_path)

    # Verify the record in the database
    cursor.execute("SELECT name FROM model_meta WHERE uuid = %s", (model_uuid,))
    result = cursor.fetchone()
    if result and result[0] == folder_name:
        print(f"Record inserted into database with UUID: {model_uuid}")
    else:
        print(f"Failed to insert record into database with UUID: {model_uuid}")

    # Verify the file in S3
    s3_client = boto3.client('s3')
    try:
        s3_client.head_object(Bucket=s3_bucket, Key=model_uuid)
        print(f"File uploaded to S3 with UUID: {model_uuid}")
    except Exception as e:
        print(f"Failed to upload file to S3 with UUID: {model_uuid}")

    # Delete the record from the database
    cursor.execute("DELETE FROM model_meta WHERE uuid = %s", (model_uuid,))
    connection.commit()
    print(f"Record deleted from database with UUID: {model_uuid}")

    # Delete the file from S3
    delete_from_s3(s3_bucket, model_uuid)
    print(f"File deleted from S3 with UUID: {model_uuid}")

    # Close the cursor and connection
    cursor.close()
    connection.close()

# Input parameters
model_folder = "/home/ec2-user/environment/llama-main/llama-2-7b" # Replace with your model folder path
s3_bucket = "sliced-models" # Replace with your S3 bucket name
def read_db_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(' = ')
            if key == "db_port":
                config[key] = int(value)
            else:
                config[key] = value
    return config

# Usage example
file_path = "/home/ec2-user/environment/infastructure/db/connection-creds.txt"
config = read_db_config(file_path)
db_host = config['db_host']
db_port = config['db_port']
db_user = config['db_user']
db_password = config['db_password']

# Run the script
main(model_folder, s3_bucket, db_host, db_port, db_user, db_password)
