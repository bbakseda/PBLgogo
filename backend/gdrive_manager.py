import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

class GDriveManager:
    """
    구글 드라이브(Google Drive) API를 활용하여 지정된 공유 폴더 내에서
    벡터 데이터베이스 파일(index.faiss, index.pkl)을 업로드하고 다운로드하는 매니저 클래스입니다.
    기존 GCSManager와 호환성을 갖추도록 인터페이스를 설계했습니다.
    """
    def __init__(self, folder_id=None, credentials_path=None, credentials_info=None):
        self.folder_id = folder_id
        self.service = None
        
        # 구글 드라이브 API 접근을 위한 스코프 지정
        scopes = ['https://www.googleapis.com/auth/drive']
        credentials = None
        
        # 1. Credentials 딕셔너리 정보가 제공된 경우 (Streamlit 세션 등)
        if credentials_info:
            try:
                credentials = service_account.Credentials.from_service_account_info(credentials_info, scopes=scopes)
            except Exception as e:
                print(f"Error loading GDrive credentials from info: {e}")
        # 2. 자격증명 파일(.json) 경로가 제공된 경우
        elif credentials_path and os.path.exists(credentials_path):
            try:
                credentials = service_account.Credentials.from_service_account_file(credentials_path, scopes=scopes)
            except Exception as e:
                print(f"Error loading GDrive credentials from file: {e}")
        # 3. 환경 변수에서 감지된 경우
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), scopes=scopes
                )
            except Exception as e:
                print(f"Error loading default GDrive credentials: {e}")

        if credentials:
            try:
                # 구글 드라이브 API 서비스 클라이언트 빌드
                self.service = build('drive', 'v3', credentials=credentials)
            except Exception as e:
                print(f"Error building GDrive API client: {e}")

    def is_connected(self):
        """구글 드라이브 서비스 클라이언트 및 공유 폴더 ID가 유효한지 확인합니다."""
        return self.service is not None and bool(self.folder_id)

    def list_files(self):
        """지정된 공유 폴더 내부의 활성 파일 목록(이름)을 반환합니다."""
        if not self.is_connected():
            return []
        try:
            query = f"'{self.folder_id}' in parents and trashed = false"
            results = self.service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
            files = results.get('files', [])
            return [f['name'] for f in files]
        except Exception as e:
            print(f"Error listing GDrive folder files: {e}")
            return []

    def upload_file(self, local_file_path, destination_name=None):
        """로컬 파일을 구글 드라이브 지정 폴더에 업로드합니다. 동일 파일명 존재 시 덮어씁니다."""
        if not self.is_connected():
            raise ValueError("GDrive Manager is not properly connected.")
            
        if not destination_name:
            destination_name = os.path.basename(local_file_path)
            
        try:
            # 기존 동일 명칭의 파일 존재 유무 탐색
            query = f"name = '{destination_name}' and '{self.folder_id}' in parents and trashed = false"
            results = self.service.files().list(q=query, spaces='drive', fields='files(id)').execute()
            files = results.get('files', [])
            
            media = MediaFileUpload(local_file_path, resumable=True)
            
            if files:
                # 덮어쓰기 (Update)
                file_id = files[0]['id']
                file = self.service.files().update(
                    fileId=file_id,
                    media_body=media
                ).execute()
                print(f"Successfully updated GDrive file: {destination_name} (ID: {file.get('id')})")
            else:
                # 신규 파일 업로드 (Create)
                file_metadata = {
                    'name': destination_name,
                    'parents': [self.folder_id]
                }
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                print(f"Successfully created GDrive file: {destination_name} (ID: {file.get('id')})")
            return True
        except Exception as e:
            print(f"Error uploading {destination_name} to GDrive: {e}")
            return False

    def download_file(self, blob_name, destination_folder):
        """구글 드라이브 폴더 내의 특정 파일을 다운로드하여 로컬 폴더에 저장합니다."""
        if not self.is_connected():
            raise ValueError("GDrive Manager is not properly connected.")
            
        # blob_name에서 파일 명칭 추출
        file_name = os.path.basename(blob_name)
        os.makedirs(destination_folder, exist_ok=True)
        dest_path = os.path.join(destination_folder, file_name)
        
        try:
            # 파일 ID 검색
            query = f"name = '{file_name}' and '{self.folder_id}' in parents and trashed = false"
            results = self.service.files().list(q=query, spaces='drive', fields='files(id)').execute()
            files = results.get('files', [])
            
            if not files:
                print(f"GDrive file not found: {file_name}")
                return None
                
            file_id = files[0]['id']
            request = self.service.files().get_media(fileId=file_id)
            
            with io.FileIO(dest_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    
            print(f"Downloaded {file_name} from GDrive to {dest_path}")
            return dest_path
        except Exception as e:
            print(f"Error downloading {file_name} from GDrive: {e}")
            return None
