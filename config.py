import os
import sys

# 🚨 리눅스/클라우드 배포 환경 한글 ASCII 인코딩 크래시 방지용 UTF-8 로케일 강제 이식
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "ko_KR.UTF-8"
os.environ["LC_ALL"] = "ko_KR.UTF-8"

import streamlit as st
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Streamlit secrets 및 환경변수 하이브리드 파싱 헬퍼 (로컬/배포 공용)
def get_config_val(key, default=""):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except:
        pass
    return os.getenv(key, default)

# Ollama 설정
OLLAMA_HOST = get_config_val("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = get_config_val("OLLAMA_MODEL", "gemma4:e4b")

# Google Cloud Storage 설정
GCS_BUCKET_NAME = get_config_val("GCS_BUCKET_NAME", "")
GCP_PROJECT_ID = get_config_val("GCP_PROJECT_ID", "")
GOOGLE_APPLICATION_CREDENTIALS = get_config_val("GOOGLE_APPLICATION_CREDENTIALS", "")
GEMINI_API_KEY = get_config_val("GEMINI_API_KEY", "")
MY_LOCAL_OLLAMA_URL = get_config_val("MY_LOCAL_OLLAMA_URL", "")
SERVER_OWNER_NAME = get_config_val("SERVER_OWNER_NAME", "박세훈 교사")

# --- 경로 설정 변경 (한글 사용자명 회피 및 리눅스 배포 환경 분기) ---
# Windows 환경은 영문 공용폴더를 활용하고, 리눅스(Streamlit Cloud) 환경은 프로젝트 내부 임시 경로를 이용합니다.
if os.name == 'nt':
    BASE_DATA_DIR = "C:/Users/Public/.elementary_assistant"
else:
    BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gcs_temp_data")

DATA_DIR = os.path.join(BASE_DATA_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DATA_DIR, "vector_store")

# 폴더 생성 보장
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
