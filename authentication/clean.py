import boto3

# Initialize clients for S3, KMS, and Cognito
s3_client = boto3.client('s3')
kms_client = boto3.client('kms')
cognito_client = boto3.client('cognito-idp')

# 1. Delete the encrypted file from the S3 bucket
bucket_name = "my-test-bucket"  # Replace with your bucket name
s3_client.delete_object(Bucket=bucket_name, Key='hello_encrypted_user_encrypted.txt')

# 2. Delete the text file from the S3 bucket
s3_client.delete_object(Bucket=bucket_name, Key='hello_encrypted_user.txt')

# 3. Delete the S3 bucket
s3_client.delete_bucket(Bucket=bucket_name)
print(f"Deleted S3 bucket: {bucket_name}")

# 4. Delete the KMS key
key_id = 'your_kms_key_id'  # Replace with your KMS key ID
kms_client.schedule_key_deletion(
    KeyId=key_id,
    PendingWindowInDays=7  # The minimum days to deletion is 7
)
print(f"Scheduled KMS key deletion: {key_id}")

# 5. Delete the user from the Cognito User Pool
user_pool_id = 'your_user_pool_id'  # Replace with your user pool ID
username = 'test@example.com'  # Replace with your username
cognito_client.admin_delete_user(
    UserPoolId=user_pool_id,
    Username=username
)
print(f"Deleted user: {username}")

# 6. Delete the Cognito User Pool
cognito_client.delete_user_pool(
    UserPoolId=user_pool_id
)
print(f"Deleted Cognito User Pool: {user_pool_id}")
