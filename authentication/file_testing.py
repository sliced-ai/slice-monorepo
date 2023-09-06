import boto3
import warnings

# Initialize Cognito and other services
region_name = 'us-west-2'
cognito_client = boto3.client('cognito-idp', region_name=region_name)
s3 = boto3.client('s3', region_name=region_name)
kms = boto3.client('kms', region_name=region_name)

# Provided user pool ID, app client ID, and username
user_pool_id = 'us-west-2_IEaead6V8'
client_id = '1lgu3a3gu9ftvsriev1pk7hs8m'
username = 'test@example.com'

# Authenticate the user and get the ID token
response = cognito_client.admin_initiate_auth(
    UserPoolId=user_pool_id,
    ClientId=client_id,
    AuthFlow='ADMIN_NO_SRP_AUTH',
    AuthParameters={
        'USERNAME': username,
        'PASSWORD': 'New@1234'  # Replace with the user's actual password
    }
)
id_token = response['AuthenticationResult']['IdToken']

# Create a KMS key (or use an existing one)
key_response = kms.create_key(
    Description='key-for-encrypting-user-data'
)
key_id = key_response['KeyMetadata']['KeyId']

# Create a text file and encrypt it using KMS
plaintext = "Hello encrypted user"
encrypt_response = kms.encrypt(
    KeyId=key_id,
    Plaintext=plaintext
)
encrypted_text = encrypt_response['CiphertextBlob']

# Save encrypted file to S3 bucket
bucket_name = 'slicetestingencryption'
s3.put_object(Bucket=bucket_name, Key='hello_encrypted_user_encrypted.txt', Body=encrypted_text)

# Load the encrypted file from S3
s3_response = s3.get_object(Bucket=bucket_name, Key='hello_encrypted_user_encrypted.txt')
encrypted_text_from_s3 = s3_response['Body'].read()

# Decrypt the file using KMS
decrypt_response = kms.decrypt(
    KeyId=key_id,
    CiphertextBlob=encrypted_text_from_s3
)
decrypted_text = decrypt_response['Plaintext'].decode('utf-8')

# Save the decrypted file back to S3 (optional)
s3.put_object(Bucket=bucket_name, Key='hello_encrypted_user_decrypted.txt', Body=decrypted_text)

# Print decrypted text to verify
print(f"Decrypted text: {decrypted_text}")

# Note: For production, you would typically use the user's credentials to assume a role that has permissions to use the KMS key.
