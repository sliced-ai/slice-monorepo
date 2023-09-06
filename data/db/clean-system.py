import mysql.connector
import boto3

# Input parameters
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



# Flags to determine what to delete
delete_s3_bucket_data = True
delete_db_table_data = True
delete_db_tables = False

def delete_bucket_data(bucket):
    s3_client = boto3.client('s3')
    s3_client.delete_objects(
        Bucket=bucket,
        Delete={'Objects': [{'Key': obj['Key']} for obj in s3_client.list_objects_v2(Bucket=bucket)['Contents']]},
    )

def delete_table_data(connection, table):
    cursor = connection.cursor()
    cursor.execute(f"DELETE FROM {table}")
    connection.commit()
    cursor.close()

def delete_table(connection, table):
    cursor = connection.cursor()
    cursor.execute(f"DROP TABLE {table}")
    connection.commit()
    cursor.close()

def main():
    # Connect to the database
    connection = mysql.connector.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database="slice",
    )

    # Delete S3 bucket data
    if delete_s3_bucket_data:
        delete_bucket_data(s3_bucket)
        print(f"Deleted data from S3 bucket {s3_bucket}")

    # Delete database table data
    if delete_db_table_data:
        tables = ["s3_data", "model_weights", "model_meta", "text_data"]
        for table in tables:
            delete_table_data(connection, table)
            print(f"Deleted data from table {table}")

    # Delete database tables
    if delete_db_tables:
        tables = ["s3_data", "model_weights", "model_meta", "text_data"]
        for table in tables:
            delete_table(connection, table)
            print(f"Deleted table {table}")

    # Close the connection
    connection.close()

# Run the script
main()
