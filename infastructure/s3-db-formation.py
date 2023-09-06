import mysql.connector

# Database connection parameters
# Database connection parameters
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
host = config['db_host']
port = config['db_port']
user = config['db_user']
password = config['db_password']
database = "slice"

# Establish a connection to the server (without specifying a database)
connection = mysql.connector.connect(
    host=host,
    port=port,
    user=user,
    password=password,
)
cursor = connection.cursor()

# Create the database if it does not exist
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")

# Close the initial connection
cursor.close()
connection.close()

# Establish a new connection to the created database
connection = mysql.connector.connect(
    host=host,
    port=port,
    user=user,
    password=password,
    database=database,
)
cursor = connection.cursor()

# Create the tables and their columns

# s3-data table
cursor.execute("""
CREATE TABLE IF NOT EXISTS s3_data (
    uuid VARCHAR(36) PRIMARY KEY,
    type VARCHAR(255),
    bucket VARCHAR(255)
);
""")

# model-weights table
cursor.execute("""
CREATE TABLE IF NOT EXISTS model_weights (
    uuid VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255),
    total_shard INT,
    shard_number INT
);
""")

# model-meta table
cursor.execute("""
CREATE TABLE IF NOT EXISTS model_meta (
    uuid VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255)
);
""")

# text-data table
cursor.execute("""
CREATE TABLE IF NOT EXISTS text_data (
    uuid VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255),
    total_shard INT,
    shard_number INT
);
""")

# Close the cursor and connection
cursor.close()
connection.close()

print("Database and tables created successfully.")
