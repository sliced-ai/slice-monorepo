import os
import boto3
import uuid
import logging
logging.basicConfig(level=logging.INFO)

class DataMover:
    
    def __init__(self, log_folder: str):
        generated_uuid = uuid.uuid4()
        uuid_string = str(generated_uuid)
        self.run_uuid = uuid_string
        self.log_folder = log_folder
        self.s3_client = boto3.client('s3')

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        log_file_path = os.path.join(self.log_folder, f"{self.run_uuid}.log")
        file_handler = logging.FileHandler(log_file_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def download_json_from_s3(self, bucket: str, key: str, download_path: str) -> None:
        try:
            self.s3_client.download_file(bucket, key, download_path)
            self.logger.info(f"Successfully downloaded {key} from {bucket} to {download_path}")
        except Exception as e:
            self.logger.error(f"An error occurred while downloading the file from S3: {e}")
            raise

    
    ################################
    def move_folder_to_s3(self, local_folder_path, bucket_name, s3_folder):
        for root, dirs, files in os.walk(local_folder_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_folder_path)
                s3_path = os.path.join(s3_folder, self.run_uuid, relative_path)
                try:
                    self.s3_client.upload_file(local_file_path, bucket_name, s3_path)
                    self.logger.info(f'Successfully moved {local_file_path} to {bucket_name}/{s3_path}')
                except Exception as e:
                    self.logger.error(f'An error occurred: {e}')
