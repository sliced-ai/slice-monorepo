import boto3

# Specify the region
region_name = 'us-west-2'  # Replace with the region you are using

client = boto3.client('cognito-idp', region_name=region_name)

# Create User Pool
response = client.create_user_pool(
    PoolName='MyTestUserPool',
    AutoVerifiedAttributes=['email']
)

user_pool_id = response['UserPool']['Id']
print(f"User pool created with ID: {user_pool_id}")

# Create an app client with auth flow enabled
app_response = client.create_user_pool_client(
    UserPoolId=user_pool_id,
    ClientName='MyTestAppClient',
    GenerateSecret=False,
    ExplicitAuthFlows=[
        'ADMIN_NO_SRP_AUTH'
    ]
)

client_id = app_response['UserPoolClient']['ClientId']
print(f"App client created with ID: {client_id}")

# Create User with initial password
response = client.admin_create_user(
    UserPoolId=user_pool_id,
    Username='test@example.com',
    TemporaryPassword='Temp@1234',  # Setting the initial password
    UserAttributes=[
        {
            'Name': 'email',
            'Value': 'test@example.com'
        },
    ],
    MessageAction='SUPPRESS'
)


print(f"User created: {response['User']['Username']}")

# Initial authentication to get session
response = client.admin_initiate_auth(
    UserPoolId=user_pool_id,
    ClientId=client_id,
    AuthFlow='ADMIN_NO_SRP_AUTH',
    AuthParameters={
        'USERNAME': 'test@example.com',
        'PASSWORD': 'Temp@1234'  # Initial password
    }
)

# Extract session details
session = response['Session']

# Change the password
new_password_response = client.admin_respond_to_auth_challenge(
    UserPoolId=user_pool_id,
    ClientId=client_id,
    ChallengeName='NEW_PASSWORD_REQUIRED',
    ChallengeResponses={
        'USERNAME': 'test@example.com',
        'NEW_PASSWORD': 'New@1234'  # New password
    },
    Session=session
)

# Extracting new tokens
id_token = new_password_response['AuthenticationResult']['IdToken']
