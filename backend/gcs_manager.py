import os
import json
from google.cloud import storage
from google.oauth2 import service_account

class GCSManager:
    def __init__(self, bucket_name=None, credentials_path=None, credentials_info=None):
        self.bucket_name = bucket_name
        self.client = None
        self.bucket = None
        
        # 1. credentials_info(dict 형태)가 제공된 경우 (Streamlit 업로드 등)
        if credentials_info:
            try:
                credentials = service_account.Credentials.from_service_account_info(credentials_info)
                self.client = storage.Client(credentials=credentials)
            except Exception as e:
                print(f"Error loading credentials from info: {e}")
        # 2. credentials_path가 지정된 경우
        elif credentials_path and os.path.exists(credentials_path):
            try:
                self.client = storage.Client.from_service_account_json(credentials_path)
            except Exception as e:
                print(f"Error loading credentials from path: {e}")
        # 3. 환경 변수에 GOOGLE_APPLICATION_CREDENTIALS가 설정된 경우
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                self.client = storage.Client()
            except Exception as e:
                print(f"Error loading default credentials: {e}")
                
        if self.client and self.bucket_name:
            try:
                self.bucket = self.client.bucket(self.bucket_name)
            except Exception as e:
                print(f"Error getting bucket: {e}")

    def is_connected(self):
        return self.client is not None and self.bucket is not None

    def list_files(self):
        """버킷 내의 모든 파일 목록을 반환합니다."""
        if not self.bucket:
            return []
        try:
            blobs = self.client.list_blobs(self.bucket_name)
            return [blob.name for blob in blobs]
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def download_file(self, blob_name, destination_folder):
        """특정 파일을 로컬 디렉토리로 다운로드합니다."""
        if not self.bucket:
            raise ValueError("GCS Bucket is not initialized.")
        
        os.makedirs(destination_folder, exist_ok=True)
        destination_path = os.path.join(destination_folder, os.path.basename(blob_name))
        
        try:
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(destination_path)
            return destination_path
        except Exception as e:
            print(f"Error downloading {blob_name}: {e}")
            return None

    def upload_file(self, local_file_path, destination_blob_name):
        """로컬 파일을 GCS 버킷으로 업로드합니다."""
        if not self.bucket:
            raise ValueError("GCS Bucket is not initialized.")
            
        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(local_file_path)
            return True
        except Exception as e:
            print(f"Error uploading {local_file_path}: {e}")
            return False
            
    def sync_all_files(self, destination_folder):
        """버킷 내의 모든 파일을 로컬 폴더와 동기화(다운로드)합니다."""
        files = self.list_files()
        downloaded = []
        for file in files:
            # 폴더 구조가 아닌 파일만 다운로드 (끝이 '/'로 끝나지 않는 것)
            if not file.endswith('/'):
                path = self.download_file(file, destination_folder)
                if path:
                    downloaded.append(path)
        return downloaded
