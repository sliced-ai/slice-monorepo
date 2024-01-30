
import os
import uuid
import tarfile
import mysql.connector
import boto3


def upload_to_s3(bucket, s3_key, local_path):
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_path, bucket, s3_key)

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

    # Close the cursor and connection
    cursor.close()
    connection.close()

    print(f"Model files tarred and uploaded to S3 with UUID: {model_uuid}")

def insert_into_db(cursor, table, values):
    if table == "model_weights":
        query = "INSERT INTO model_weights (uuid, name, shard_number, total_shard) VALUES (%s, %s, %s, %s)"
    elif table == "s3_data":
        query = "INSERT INTO s3_data (uuid, type, bucket) VALUES (%s, %s, %s)"
    cursor.execute(query, values)

def process_pth_files(model_folder, s3_bucket, cursor):
    # Get all the .pth files and sort them
    pth_files = [f for f in os.listdir(model_folder) if f.endswith('.pth')]
    pth_files.sort()

    total_shards = len(pth_files)
    for i, pth_file in enumerate(pth_files):
        # Generate UUID for the tar file
        pth_uuid = str(uuid.uuid4())

        # Create tar file
        tar_path = os.path.join(model_folder, pth_uuid)
        with tarfile.open(tar_path, 'w') as tarf:
            file_path = os.path.join(model_folder, pth_file)
            tarf.add(file_path, arcname=os.path.basename(file_path))

        # Upload to S3
        upload_to_s3(s3_bucket, pth_uuid, tar_path)

        # Insert into model_weights table
        folder_name = os.path.basename(model_folder)
        shard = i + 1
        insert_into_db(cursor, "model_weights", (pth_uuid, folder_name, shard, total_shards))

        # Insert into s3_data table
        insert_into_db(cursor, "s3_data", (pth_uuid, "model_weights", s3_bucket))


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

connection = mysql.connector.connect(
    host=db_host,
    port=db_port,
    user=db_user,
    password=db_password,
    database="slice",
)
cursor = connection.cursor()
process_pth_files(model_folder, s3_bucket, cursor)
cursor.close()
connection.close()


import re
import os

def delete_uuid_files(directory):
    # Define a regular expression pattern for UUIDs
    uuid_pattern = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
    
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        # If the filename matches the UUID pattern, delete the file
        if uuid_pattern.match(filename):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

# Usage example:
delete_uuid_files(model_folder)