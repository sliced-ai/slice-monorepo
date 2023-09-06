import mysql.connector

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
database = "slice_test_db"  # Note: Database names cannot have hyphens

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

# 1. Create a table with 3 columns
create_table_query = """
CREATE TABLE IF NOT EXISTS example_table (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    age INT,
    address VARCHAR(255)
);
"""
cursor.execute(create_table_query)

# 2. Add a row to that table
insert_query = "INSERT INTO example_table (name, age, address) VALUES (%s, %s, %s)"
data = ("John Doe", 28, "123 Main St")
cursor.execute(insert_query, data)
connection.commit()

# Get the last inserted id
row_id = cursor.lastrowid

# 3. Read that row
select_query = f"SELECT * FROM example_table WHERE id = {row_id}"
cursor.execute(select_query)
row = cursor.fetchone()
print("Inserted row:", row)

# 4. Delete that row
delete_query = f"DELETE FROM example_table WHERE id = {row_id}"
cursor.execute(delete_query)
connection.commit()

# Verify the row has been deleted
cursor.execute(select_query)
row = cursor.fetchone()
print("Row after deletion:", row)

# 5. Delete that table
drop_table_query = "DROP TABLE example_table"
cursor.execute(drop_table_query)

# Close the cursor and connection
cursor.close()
connection.close()
