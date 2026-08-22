import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Ollama 설정
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

# Google Cloud Storage 설정
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MY_LOCAL_OLLAMA_URL = os.getenv("MY_LOCAL_OLLAMA_URL", "")
SERVER_OWNER_NAME = os.getenv("SERVER_OWNER_NAME", "박세훈 교사")

# --- 경로 설정 변경 (한글 사용자명으로 인한 FAISS Illegal byte sequence 완벽 회피) ---
# Windows 사용자 이름에 한글(예: 박세훈)이 포함된 경우, 홈 디렉토리 경로에도 한글이 들어가 에러가 납니다.
# 따라서 100% 영문으로만 이루어지고 쓰기 권한이 자유로운 공용 폴더(C:/Users/Public)를 임시/영구 데이터 저장소로 사용합니다.
BASE_DATA_DIR = "C:/Users/Public/.elementary_assistant"

DATA_DIR = os.path.join(BASE_DATA_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DATA_DIR, "vector_store")

# 폴더 생성 보장
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
