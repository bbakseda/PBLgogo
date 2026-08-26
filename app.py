import streamlit as st

# 🚨 st.set_page_config는 반드시 그 어떤 Streamlit 명령보다 가장 먼저 실행되어야 합니다! (크래시 방어)
st.set_page_config(
    page_title="초등 프로젝트 수업 계획 및 평가 비서",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import sys
import locale

# 🚨 C-Extension 및 Google SDK 인코딩 오류 원천 차단 (런타임 로케일 선점)
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

# 🚨 httpx / httpcore의 윈도우 프록시 자동 감지 버그(WinError 10049) 방지를 위해 관련 환경변수 강제 비우기
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        pass

import json
import requests
from config import OLLAMA_HOST, OLLAMA_MODEL, DATA_DIR, VECTOR_DB_DIR, GCS_BUCKET_NAME, GD_FOLDER_ID, GOOGLE_APPLICATION_CREDENTIALS, GEMINI_API_KEY, MY_LOCAL_OLLAMA_URL, ADMIN_PASSWORD
from backend.gcs_manager import GCSManager
from backend.gdrive_manager import GDriveManager
from backend.vector_store import VectorStoreManager
from backend.rag_chain import RAGChainManager
from backend.pdf_generator import markdown_to_pdf_bytes

# GPU 가속 관련 변수 기본값 선언 (웹 배포 서버와 로컬 환경의 PyTorch 의존성 격리)
cuda_available = False
gpu_name = "None"
run_indexing = None

try:
    import torch
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    from gpu_indexer import run_indexing
except Exception:
    pass

# 임베딩 모델 캐싱
@st.cache_resource
def load_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'}
    )

from backend.queue_manager import GlobalQueueManager
import uuid

# 🚨 전역 대기열 매니저 획득 (모든 스레드 세션이 공유하는 싱글톤 객체)
@st.cache_resource
def get_queue_manager():
    return GlobalQueueManager()

queue_manager = get_queue_manager()

# 🚨 세션 고유 식별자 생성 및 대기열 등록/하트비트 가동
if "session_uuid" not in st.session_state:
    st.session_state.session_uuid = str(uuid.uuid4())

session_id = st.session_state.session_uuid
my_animal_name = queue_manager.register_user(session_id)
queue_manager.keep_alive(session_id)

# 세션 상태 초기화
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if "gcs_manager" not in st.session_state:
    if GCS_BUCKET_NAME and (GOOGLE_APPLICATION_CREDENTIALS or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")):
        creds_path = GOOGLE_APPLICATION_CREDENTIALS
        if creds_path and not os.path.isabs(creds_path):
            creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), creds_path)
            
        st.session_state.gcs_manager = GCSManager(
            bucket_name=GCS_BUCKET_NAME,
            credentials_path=creds_path if os.path.exists(creds_path) else None
        )
    else:
        st.session_state.gcs_manager = None

if "gdrive_manager" not in st.session_state:
    if GD_FOLDER_ID and (GOOGLE_APPLICATION_CREDENTIALS or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")):
        creds_path = GOOGLE_APPLICATION_CREDENTIALS
        if creds_path and not os.path.isabs(creds_path):
            creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), creds_path)
            
        st.session_state.gdrive_manager = GDriveManager(
            folder_id=GD_FOLDER_ID,
            credentials_path=creds_path if os.path.exists(creds_path) else None
        )
    else:
        st.session_state.gdrive_manager = None

if "vector_manager" not in st.session_state:
    # 🚨 교사용 AI 엔진 초기화 과정을 백분율(%)과 단계별 상태 메시지로 시각화하여 사용자의 지루함 해결
    init_progress = st.progress(0.0)
    init_status = st.empty()
    
    # 1단계: 교육 자료 동기화 (15%)
    init_status.info("⏳ [1/4] RAG 교육 참고 자료를 연산 디렉토리로 동기화하는 중... (15%)")
    init_progress.progress(0.15)
    
    import shutil
    src_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if os.path.exists(src_data_dir):
        for file_name in os.listdir(src_data_dir):
            src_file = os.path.join(src_data_dir, file_name)
            dst_file = os.path.join(DATA_DIR, file_name)
            if os.path.isfile(src_file):
                # 파일이 없거나 최신 파일인 경우 자동 복사
                if not os.path.exists(dst_file) or os.path.getmtime(src_file) > os.path.getmtime(dst_file):
                    shutil.copy(src_file, dst_file)
        
    # 2단계: 문장 임베딩 모델 로드 (50%)
    init_status.info("⏳ [2/4] 초등 교육 특화 한국어 문장 임베딩 모델(ko-sroberta)을 메모리에 적재하는 중... (50%)")
    init_progress.progress(0.50)
    embeddings = load_embeddings()
    
    # 3단계: 벡터스토어 로드 및 감지 (75%)
    init_status.info("⏳ [3/4] 참고 문서 기반 벡터 데이터베이스(FAISS)를 로드 및 대조하는 중... (75%)")
    init_progress.progress(0.75)
    st.session_state.vector_manager = VectorStoreManager(DATA_DIR, VECTOR_DB_DIR, embeddings=embeddings)
    
    # 1. 저장된 벡터 스토어 로드
    vs = st.session_state.vector_manager.load_vector_store()
    
    # 2. 로컬 data/ 내의 파일 개수 파악
    num_local_files = len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]) if os.path.exists(DATA_DIR) else 0
    
    # 3. 디스크에 빌드된 벡터스토어 파일이 완전히 부재할 때만 (최초 1회) 자동 빌드 실행
    if not vs:
        def build_progress_callback(percent, status_text):
            init_status.info(status_text)
            init_progress.progress(percent)
        vs = st.session_state.vector_manager.build_vector_store(progress_callback=build_progress_callback)
        
    # 4단계: 하이브리드 RAG 체인 오케스트레이션 (95%)
    init_status.info("⏳ [4/4] 로컬/클라우드 하이브리드 RAG 교수 설계 체인을 활성화하는 중... (95%)")
    init_progress.progress(0.95)
    
    # 🚨 OLLAMA_HOST 강제 자가정화 (Secrets 설정 오염 및 윈도우 WinError 10049 소켓 충돌 방지)
    clean_ollama_host = OLLAMA_HOST
    if "localhost" in clean_ollama_host:
        clean_ollama_host = clean_ollama_host.replace("localhost", "127.0.0.1")
    if os.name == 'nt':
        clean_ollama_host = "http://127.0.0.1:11434"
        
    import config
    st.session_state.rag_manager = RAGChainManager(clean_ollama_host, OLLAMA_MODEL, vs, gemini_api_key=config.GEMINI_API_KEY)
    
    # 완료 (100% 및 청소)
    init_status.success("✅ 교사용 RAG AI 비서 엔진 로딩 완료! (100%)")
    init_progress.progress(1.0)
    import time
    time.sleep(1)
    init_status.empty()
    init_progress.empty()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "temp_plan" not in st.session_state:
    st.session_state.temp_plan = ""
if "temp_sim_report" not in st.session_state:
    st.session_state.temp_sim_report = ""
if "temp_ref_docs" not in st.session_state:
    st.session_state.temp_ref_docs = []
if "temp_report" not in st.session_state:
    st.session_state.temp_report = ""
if "temp_report_ref_docs" not in st.session_state:
    st.session_state.temp_report_ref_docs = []

# 고급스러운 다크/블루 톤 테마 CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body {
        font-family: 'Outfit', sans-serif;
    }
    
    /* 타이틀 그라데이션 */
    .title-gradient {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA, #3B82F6, #1D4ED8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
        padding-top: 10px;
    }
    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* 상태 상자 */
    .status-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 15px;
    }
    
    /* 탭 스타일 조정 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1E293B;
        border-radius: 8px 8px 0px 0px;
        border: 1px solid #334155;
        color: #94A3B8;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB;
        color: white;
        border: 1px solid #2563EB;
    }

    /* 📱 스마트폰 및 태블릿용 반응형 미디어 쿼리 */
    @media only screen and (max-width: 768px) {
        .title-gradient {
            font-size: 1.6rem !important;
            padding-top: 5px !important;
            margin-bottom: 2px !important;
            line-height: 1.3;
        }
        .subtitle {
            font-size: 0.85rem !important;
            margin-bottom: 15px !important;
        }
        /* 탭 메뉴가 가로 밖으로 밀려 깨지지 않도록 자동 패딩 축소 */
        .stTabs [data-baseweb="tab"] {
            height: 40px !important;
            padding: 5px 12px !important;
            font-size: 0.82rem !important;
        }
        /* 모바일에서는 컬럼 구조를 세로 방향으로 자연스럽게 정렬 */
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 auto !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# 메인 타이틀
st.markdown("<div class='title-gradient'>🏫 초등 프로젝트 수업 계획 및 평가 비서</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>구글 클라우드 RAG 연동 및 Gemma-4 교육 설계 인공지능</div>", unsafe_allow_html=True)

# 시스템 상태 체크
ollama_connected = False
try:
    # 1. 로컬 기본 Ollama 핑
    res = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
    if res.status_code == 200:
        ollama_connected = True
except:
    pass

# 2. 로컬은 연결 불가하지만 원격 노출 ngrok 터널링 주소가 있으면 추가 핑
if not ollama_connected and MY_LOCAL_OLLAMA_URL:
    try:
        res = requests.get(f"{MY_LOCAL_OLLAMA_URL.rstrip('/')}/api/tags", timeout=2)
        if res.status_code == 200:
            ollama_connected = True
    except:
        pass

# 3. AI 구동 가능 여부 (Ollama 연결 또는 Gemini API 중 최소 하나만 구동되면 성공)
ai_service_available = ollama_connected or bool(GEMINI_API_KEY)

# 사이드바
with st.sidebar:
    if st.session_state.is_admin:
        # --- 관리자 모드 사이드바 ---
        st.markdown("### ⚙️ [관리자 모드] 저장소 설정")
        
        storage_mode = st.radio(
            "사용할 원격 저장소 선택",
            ["구글 드라이브 (GDrive)", "구글 클라우드 (GCS)", "로컬 전용"],
            index=0 if GD_FOLDER_ID else (1 if GCS_BUCKET_NAME else 2)
        )
        
        gcs_active = st.session_state.gcs_manager and st.session_state.gcs_manager.is_connected()
        gdrive_active = st.session_state.gdrive_manager and st.session_state.gdrive_manager.is_connected()
        
        if storage_mode == "구글 드라이브 (GDrive)":
            if gdrive_active:
                st.success(f"📁 **구글 드라이브 연동 완료**\n- 폴더: `{GD_FOLDER_ID[:15]}...`")
            else:
                st.info("💻 **로컬 기본 자료 모드**\n- 구글 드라이브 설정이 유효하지 않습니다.")
                
            with st.expander("⚙️ 구글 드라이브 상세 연결 설정"):
                st.caption("공유 폴더를 연결하려면 자격증명 파일(.json)과 폴더 ID를 지정해 주세요.")
                gd_key = st.file_uploader("GCP 서비스 계정 JSON 키 업로드 (GDrive)", type=["json"], key="gd_key_uploader")
                gd_folder = st.text_input("구글 드라이브 폴더 ID", value=GD_FOLDER_ID if GD_FOLDER_ID else "", key="gd_folder_input")
                if gd_key and gd_folder:
                    try:
                        key_data = json.load(gd_key)
                        st.session_state.gdrive_manager = GDriveManager(
                            folder_id=gd_folder,
                            credentials_info=key_data
                        )
                        st.success("구글 드라이브가 수동 연결되었습니다!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"인증 파일 오류: {e}")
                        
        elif storage_mode == "구글 클라우드 (GCS)":
            if gcs_active:
                st.success(f"☁️ **구글 클라우드 연동 완료**\n- 버킷: `{st.session_state.gcs_manager.bucket_name}`")
            else:
                st.info("💻 **로컬 기본 자료 모드**\n- 구글 클라우드 설정이 유효하지 않습니다.")
                
            with st.expander("⚙️ 구글 클라우드 상세 연결 설정"):
                st.caption("클라우드 스토리지를 연결하려면 자격증명 파일(.json)과 버킷명을 지정해 주세요.")
                uploaded_key = st.file_uploader("GCP 서비스 계정 JSON 키 파일 업로드", type=["json"], key="gcs_key_uploader")
                manual_bucket = st.text_input("GCS 버킷명 입력", value=GCS_BUCKET_NAME if GCS_BUCKET_NAME else "", key="gcs_bucket_input")
                if uploaded_key and manual_bucket:
                    try:
                        key_data = json.load(uploaded_key)
                        st.session_state.gcs_manager = GCSManager(
                            bucket_name=manual_bucket,
                            credentials_info=key_data
                        )
                        st.success("구글 클라우드가 수동 연결되었습니다!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"인증 파일 오류: {e}")
        else:
            st.info("💻 **로컬 전용 모드**\n- 원격 동기화 없이 로컬 캐시 데이터로만 작동합니다.")
            
        st.markdown("---")
        st.markdown("### 🔑 Gemini API Key 설정")
        st.caption("교사님 PC가 꺼져 있을 때 백업 작동할 구글 클라우드 Gemini API 키를 관리합니다.")
        
        import config
        # 현재 동적 로딩된 API 키 노출
        gemini_key_input = st.text_input(
            "Gemini API Key",
            value=config.GEMINI_API_KEY,
            type="password",
            help="구글 AI 스튜디오에서 발급받은 API 키를 입력하세요."
        )
        
        if st.button("💾 Gemini API Key 적용 및 저장", key="save_gemini_key_btn", use_container_width=True):
            try:
                config.set_dynamic_config("GEMINI_API_KEY", gemini_key_input)
                # 즉각 런타임 RAG 매니저 재생성하여 새 키 바인딩
                vs = st.session_state.vector_manager.load_vector_store()
                from backend.rag_chain import RAGChainManager
                st.session_state.rag_manager = RAGChainManager(OLLAMA_HOST, OLLAMA_MODEL, vs, gemini_api_key=gemini_key_input)
                st.success("🎉 Gemini API Key가 성공적으로 업데이트 및 영구 저장되었습니다!")
                st.rerun()
            except Exception as e:
                st.error(f"API Key 반영 실패: {e}")
                
        st.markdown("---")
        st.markdown("### 📥 교육 자료 동기화")
        st.caption("원격지 저장소에 보관한 최신 문서를 다운로드하거나 빌드된 벡터 데이터베이스를 바로 로딩합니다.")
        
        if st.button("✨ 원격 자료 동기화 및 학습 적용", use_container_width=True, type="primary"):
            if storage_mode == "구글 드라이브 (GDrive)":
                if not st.session_state.gdrive_manager or not st.session_state.gdrive_manager.is_connected():
                    st.warning("연결된 구글 드라이브가 없습니다. 로컬 캐시를 사용합니다.")
                else:
                    with st.spinner("구글 드라이브에서 최신 학습 가이드 및 빌드 인덱스 가져오는 중..."):
                        try:
                            gdrive = st.session_state.gdrive_manager
                            files = gdrive.list_files()
                            downloaded_docs = []
                            downloaded_indices = 0
                            for f in files:
                                if f in ["index.faiss", "index.pkl"]:
                                    path = gdrive.download_file(f, VECTOR_DB_DIR)
                                    if path:
                                        downloaded_indices += 1
                                else:
                                    path = gdrive.download_file(f, DATA_DIR)
                                    if path:
                                        downloaded_docs.append(path)
                                        
                            if downloaded_indices == 2:
                                st.success("🎉 구글 드라이브에서 빌드 완료된 FAISS 벡터 인덱스를 다운로드하여 동기화했습니다!")
                                vs = st.session_state.vector_manager.load_vector_store()
                                if vs:
                                    st.session_state.rag_manager.set_vector_store(vs)
                                    st.success("로컬 AI 세션에 동기화 완료!")
                            else:
                                st.info(f"구글 드라이브로부터 {len(downloaded_docs)}개 문서를 다운로드했습니다.")
                                with st.spinner("로컬에서 신규 벡터 데이터베이스 빌드 중..."):
                                    vs = st.session_state.vector_manager.build_vector_store()
                                    if vs:
                                        st.session_state.rag_manager.set_vector_store(vs)
                                        st.success("로컬 빌드 및 학습 완료!")
                        except Exception as gd_sync_err:
                            st.error(f"구글 드라이브 동기화 실패: {gd_sync_err}")
                            
            elif storage_mode == "구글 클라우드 (GCS)":
                if not st.session_state.gcs_manager or not st.session_state.gcs_manager.is_connected():
                    st.warning("연결된 구글 클라우드가 없습니다. 로컬 캐시를 사용합니다.")
                else:
                    with st.spinner("구글 클라우드에서 최신 자료 가져오는 중..."):
                        downloaded = st.session_state.gcs_manager.sync_all_files(DATA_DIR)
                        st.success(f"클라우드로부터 {len(downloaded)}개 문서 다운로드 완료!")
                    with st.spinner("자료 분석 및 벡터 DB화 진행 중..."):
                        vs = st.session_state.vector_manager.build_vector_store()
                        if vs:
                            st.session_state.rag_manager.set_vector_store(vs)
                            st.success("구글 클라우드 문서 학습 완료!")
            else:
                with st.spinner("로컬 참고자료 동기화 및 학습 중..."):
                    vs = st.session_state.vector_manager.build_vector_store()
                    if vs:
                        st.session_state.rag_manager.set_vector_store(vs)
                        st.success("로컬 데이터 동기화 완료!")
                        
        st.markdown("---")
        if st.button("🔓 관리자 모드 로그아웃", key="admin_logout_btn", use_container_width=True):
            st.session_state.is_admin = False
            st.success("일반 사용자 모드로 전환되었습니다.")
            st.rerun()
            
    else:
        # --- 일반 사용자(교사) 모드 사이드바 ---
        st.markdown("### 🏫 교육 자료 저장소")
        
        # 내부적인 동기화 모드 탐색
        storage_mode = "구글 드라이브 (GDrive)" if GD_FOLDER_ID else ("구글 클라우드 (GCS)" if GCS_BUCKET_NAME else "로컬 전용")
        gcs_active = st.session_state.gcs_manager and st.session_state.gcs_manager.is_connected()
        gdrive_active = st.session_state.gdrive_manager and st.session_state.gdrive_manager.is_connected()
        
        if storage_mode == "구글 드라이브 (GDrive)" and gdrive_active:
            st.success("🟢 **교육 가이드라인 연동 중**\n- 클라우드 최신 기준 자동 반영")
        elif storage_mode == "구글 클라우드 (GCS)" and gcs_active:
            st.success("🟢 **교육 가이드라인 연동 중**\n- 구글 클라우드 버킷 연동 중")
        else:
            st.info("💻 **로컬 기본 참고서 모드**\n- 네트워크 연결이 제한된 오프라인 교안으로 작동합니다.")
            
        st.markdown("---")
        st.markdown("### 📥 교과 및 성취기준 업데이트")
        st.caption("연동된 교육자료 클라우드 저장소로부터 교육과정 성취기준 및 새로운 실습 자료를 원클릭으로 동기화합니다.")
        
        if st.button("🔄 최신 교육 자료 동기화 및 업데이트", use_container_width=True, type="primary"):
            if storage_mode == "구글 드라이브 (GDrive)" and gdrive_active:
                with st.spinner("원격 저장소로부터 최신 교육 자료 동기화 중..."):
                    try:
                        gdrive = st.session_state.gdrive_manager
                        files = gdrive.list_files()
                        downloaded_docs = []
                        downloaded_indices = 0
                        for f in files:
                            if f in ["index.faiss", "index.pkl"]:
                                path = gdrive.download_file(f, VECTOR_DB_DIR)
                                if path:
                                    downloaded_indices += 1
                            else:
                                path = gdrive.download_file(f, DATA_DIR)
                                if path:
                                    downloaded_docs.append(path)
                                    
                        if downloaded_indices == 2:
                            st.success("🎉 최신 벡터 데이터베이스를 구글 드라이브로부터 즉각 반영했습니다!")
                            vs = st.session_state.vector_manager.load_vector_store()
                            if vs:
                                st.session_state.rag_manager.set_vector_store(vs)
                                st.success("AI 비서 업데이트 완료!")
                        else:
                            with st.spinner("로컬에서 인덱스 갱신 빌드 중..."):
                                vs = st.session_state.vector_manager.build_vector_store()
                                if vs:
                                    st.session_state.rag_manager.set_vector_store(vs)
                                    st.success("AI 비서 빌드 완료!")
                    except Exception as e:
                        st.error(f"동기화 중 오류 발생: {e}")
            elif storage_mode == "구글 클라우드 (GCS)" and gcs_active:
                with st.spinner("원격 구글 클라우드 동기화 진행 중..."):
                    try:
                        downloaded = st.session_state.gcs_manager.sync_all_files(DATA_DIR)
                        vs = st.session_state.vector_manager.build_vector_store()
                        if vs:
                            st.session_state.rag_manager.set_vector_store(vs)
                            st.success(f"성공! {len(downloaded)}개 교재를 갱신 및 학습시켰습니다.")
                    except Exception as e:
                        st.error(f"동기화 중 오류: {e}")
            else:
                with st.spinner("내부 교재 목록 인덱싱 중..."):
                    vs = st.session_state.vector_manager.build_vector_store()
                    if vs:
                        st.session_state.rag_manager.set_vector_store(vs)
                        st.success("로컬 성취기준 인덱싱 완료!")
                        
        st.markdown("---")
        with st.expander("🔐 시스템 관리자 모드"):
            admin_pw = st.text_input("관리자 인증 암호 입력", type="password", key="admin_pwd_widget")
            if st.button("로그인", key="admin_login_btn", use_container_width=True):
                if admin_pw == ADMIN_PASSWORD:
                    st.session_state.is_admin = True
                    st.success("인증 성공! 관리자 설정 대시보드가 개방되었습니다.")
                    st.rerun()
                else:
                    st.error("인증 암호가 올바르지 않습니다.")

    st.markdown("---")
    st.markdown("### 🖥️ AI 엔진 및 연결 요약")
    
    # RAGChainManager에 연결된 엔진의 종류 파악
    engine_name = "기본 엔진"
    is_gemini = True
    if hasattr(st.session_state.rag_manager, "connected_engine_info"):
        engine_name = st.session_state.rag_manager.connected_engine_info
        is_gemini = st.session_state.rag_manager.is_gemini_active
        
    from config import SERVER_OWNER_NAME
    
    # 젬마4 로컬 원격 연결 성공 시 초록불 알림, 클라우드 백업 구동 시 노랑불 알림 연출
    if "gemma4" in engine_name.lower() or "로컬" in engine_name.lower():
        st.success(f"🟢 **{SERVER_OWNER_NAME}**님의 AI 서버\n- **Gemma-4 연결 성공**")
    else:
        st.info(f"🟡 **Gemini 클라우드 백업 작동**\n- {SERVER_OWNER_NAME} AI 컴퓨터가 꺼져 있어 클라우드로 자동 대체 구동됩니다.")
        
    num_files = 0
    if os.path.exists(DATA_DIR):
        num_files = len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
        
    st.markdown(f"- **현재 활성 엔진:** `{engine_name}`")
    st.markdown(f"- **학습 완료된 교과 문서:** `{num_files}`개")
    
    st.markdown("---")
    st.markdown("### 🦄 교사용 AI 대기열 현황")
    
    # 🚨 CPU 과부하 및 Throttling(제한)을 방지하기 위해 대기열 패널만 st.fragment로 부분 격리하여 실시간 자동 갱신 수행
    @st.fragment
    def render_realtime_queue():
        queue_status = queue_manager.get_queue_status(session_id)
        st.markdown(f"- 🏷️ **나의 닉네임:** `{queue_status['my_name']}(나)`")
        
        if queue_status['is_my_turn']:
            st.success("🟢 **지금 생성하실 수 있습니다!**")
        else:
            st.warning(f"⏳ **대기 중:** `내 순서: {queue_status['my_turn']}번째` (현재 집필 중: `{queue_status['active_user_name']}`)")
            
        with st.expander(f"📋 대기열 목록 ({queue_status['total_waiting']}명)"):
            for idx, name in enumerate(queue_status['queue_list']):
                if name == queue_status['my_name']:
                    st.markdown(f"**{idx+1}. {name}(나)**")
                else:
                    st.markdown(f"{idx+1}. {name}")
                    
        # 🚨 전체 스크립트를 Rerun하지 않고, 오직 대기열 패널만 10초 주기로 백그라운드 갱신하여 서버 자원 극도 절약
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=10000, limit=1000, key="queue_auto_refresh_fragment")
        
    # 격리 렌더러 호출
    render_realtime_queue()

# 탭 구성 (관리자 모드 여부에 따라 분기)
if st.session_state.is_admin:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 상세 수업 계획 및 지도안 설계", 
        "📊 수업 결과 보고 및 평가서 작성", 
        "💬 초등 교육과정 Q&A AI 비서",
        "⚡ 대용량 PDF-to-FAISS GPU 가속 인덱서"
    ])
else:
    tab1, tab2, tab3 = st.tabs([
        "📅 상세 수업 계획 및 지도안 설계", 
        "📊 수업 결과 보고 및 평가서 작성", 
        "💬 초등 교육과정 Q&A AI 비서"
    ])

# 탭 1: 상세 수업 계획 설계
with tab1:
    st.subheader("📝 프로젝트형 수업 계획서 & 세부 지도안 생성")
    st.markdown("주제와 대상을 설정하면 매 차시별 교사의 발문, 학생 활동, 세부 피드백 계획까지 담긴 상세 로드맵을 설계합니다.")
    
    col1, col2 = st.columns([1, 1.3])
    with col1:
        proj_title = st.text_input("수업 주제 / 단원명", placeholder="예: 초등 5학년 날씨와 우리 생활 (또는 엔트리로 만드는 미로 찾기)")
        learning_goals = st.text_area("핵심 학습 목표", placeholder="예: 날씨 요소와 실생활 관계를 파악하고 기상 관측 프로그램 코딩하기", height=100)
        
        c_sched, c_level = st.columns([1, 1])
        with c_sched:
            duration = st.selectbox("수업 분량 (차시)", ["1차시 (40분)", "2차시 블록", "4차시 프로젝트", "8차시 대단원 프로젝트"])
        with c_level:
            level = st.selectbox("대상 학년", ["초등 1~2학년", "초등 3~4학년", "초등 5~6학년"])
            
        model_type = st.selectbox("적용할 교수학습 모형", [
            "PBL (프로젝트 기반 학습)",
            "놀이 중심 학습 모형",
            "실험 실습 중심 모형",
            "토의 토론 학습 모형",
            "일반적인 도입-전개-정리 학습 흐름"
        ])
        
        add_reqs = st.text_area("교사의 추가 희망사항", placeholder="예: 도입 단계에 재미있는 동기유발 퀴즈를 넣어주고, 매 활동마다 3줄 요약 평가지 양식을 추가해 줘", height=80)
        generate_plan_btn = st.button("🚀 상세 수업 계획서 생성", type="primary", use_container_width=True)
        generate_both_btn = st.button("✨ 계획서 & 가상 보고서 일괄 생성", type="secondary", use_container_width=True)
        
    with col2:
        st.markdown("#### 📄 계획서 출력 및 RAG 검증")
        
        if generate_plan_btn or generate_both_btn:
            if not proj_title or not learning_goals:
                st.error("수업 주제와 핵심 학습 목표를 반드시 입력해 주세요.")
            else:
                # 🚨 대기열 검문소 가동 (동시 생성 시 1순위 독점 보장)
                status = queue_manager.get_queue_status(session_id)
                if not status['is_my_turn']:
                    st.error(f"🚨 아직 회원님의 차례가 아닙니다. 현재 `{status['active_user_name']}`님이 AI를 점유 중입니다. 회원님은 `{status['my_turn']}번째` 대기 중입니다.")
                else:
                    try:
                        # 1단계: 계획서 생성
                        with st.spinner("AI 엔진이 상세 수업계획서를 집필 중입니다..."):
                            plan_result, ref_docs = st.session_state.rag_manager.generate_study_plan(
                                project_title=proj_title,
                                learning_goals=learning_goals,
                                duration=duration,
                                level=level,
                                model_type=model_type,
                                additional_requirements=add_reqs
                            )
                            st.session_state.temp_plan = plan_result
                            st.session_state.temp_ref_docs = ref_docs
                            st.session_state.temp_sim_report = "" # 초기화
                            
                        # 2단계: 일괄 생성 시 결과 보고서도 연속 생성
                        if generate_both_btn and st.session_state.temp_plan:
                            with st.spinner("이 계획에 기반한 가상의 결과 보고서를 시뮬레이션하여 작성 중입니다..."):
                                sim_report = st.session_state.rag_manager.generate_simulated_report(
                                    project_title=proj_title,
                                    learning_goals=learning_goals,
                                    plan_content=st.session_state.temp_plan,
                                    level=level,
                                    model_type=model_type
                                )
                                st.session_state.temp_sim_report = sim_report
                    except Exception as e:
                        import traceback
                        st.error(f"생성 실패: {e}\n\n{traceback.format_exc()}")
                    finally:
                        # 🚨 연산 완료/예외 발생 무관하게 즉시 대기열 자격을 밀어주고 뒤 차례 대기자로 넘김
                        queue_manager.release_turn(session_id)
                        st.rerun()

        # Y축 겹침/증발 없는 세션 데이터 기반 렌더링
        if st.session_state.temp_plan:
            if st.session_state.temp_ref_docs and any(doc.metadata.get('source') for doc in st.session_state.temp_ref_docs):
                st.info("🔍 **[구글 클라우드 RAG 모드 검증]** 사용자의 클라우드 보관 문서 가이드라인이 분석에 직접 활용되었습니다.")
            else:
                st.info("💡 **[Gemma-4 단독 모드]** 연동된 참조 문서가 없어, AI 자체 초등 교육학 지식에 입각하여 고밀도 계획서를 생성했습니다.")
            
            # 일괄 생성 결과 렌더링 분기
            if st.session_state.temp_sim_report:
                sub_tab_plan, sub_tab_rep = st.tabs(["📅 생성된 수업 계획서", "📊 가상의 수업 결과 보고서"])
                with sub_tab_plan:
                    st.markdown(st.session_state.temp_plan)
                    
                    col_dl1, col_dl2 = st.columns([1, 1])
                    with col_dl1:
                        st.download_button(
                            label="📥 계획서 마크다운(.md) 다운로드",
                            data=st.session_state.temp_plan,
                            file_name=f"수업계획서_{proj_title.replace(' ', '_')}.md",
                            mime="text/markdown",
                            key="both_plan_md_dl",
                            use_container_width=True
                        )
                    with col_dl2:
                        try:
                            pdf_data = markdown_to_pdf_bytes(st.session_state.temp_plan, title=f"수업계획서: {proj_title}")
                            st.download_button(
                                label="📥 계획서 PDF(.pdf) 다운로드",
                                data=pdf_data,
                                file_name=f"수업계획서_{proj_title.replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                key="both_plan_pdf_dl",
                                use_container_width=True
                            )
                        except Exception as pdf_err:
                            st.warning(f"PDF 생성 실패: {pdf_err}")
                
                with sub_tab_rep:
                    st.markdown(st.session_state.temp_sim_report)
                    
                    col_dl3, col_dl4 = st.columns([1, 1])
                    with col_dl3:
                        st.download_button(
                            label="📥 가상 보고서 마크다운(.md) 다운로드",
                            data=st.session_state.temp_sim_report,
                            file_name=f"가상수업결과보고_{proj_title.replace(' ', '_')}.md",
                            mime="text/markdown",
                            key="both_rep_md_dl",
                            use_container_width=True
                        )
                    with col_dl4:
                        try:
                            pdf_data = markdown_to_pdf_bytes(st.session_state.temp_sim_report, title=f"가상수업결과보고: {proj_title}")
                            st.download_button(
                                label="📥 가상 보고서 PDF(.pdf) 다운로드",
                                data=pdf_data,
                                file_name=f"가상수업결과보고_{proj_title.replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                key="both_rep_pdf_dl",
                                use_container_width=True
                            )
                        except Exception as pdf_err:
                            st.warning(f"PDF 생성 실패: {pdf_err}")
            else:
                # 계획서 단독 출력
                st.markdown(st.session_state.temp_plan)
                col_dl1, col_dl2 = st.columns([1, 1])
                with col_dl1:
                    st.download_button(
                        label="📥 계획서 마크다운(.md) 다운로드",
                        data=st.session_state.temp_plan,
                        file_name=f"수업계획서_{proj_title.replace(' ', '_')}.md",
                        mime="text/markdown",
                        key="single_plan_md_dl",
                        use_container_width=True
                    )
                with col_dl2:
                    try:
                        pdf_data = markdown_to_pdf_bytes(st.session_state.temp_plan, title=f"수업계획서: {proj_title}")
                        st.download_button(
                            label="📥 계획서 PDF(.pdf) 다운로드",
                            data=pdf_data,
                            file_name=f"수업계획서_{proj_title.replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            key="single_plan_pdf_dl",
                            use_container_width=True
                        )
                    except Exception as pdf_err:
                        st.warning(f"PDF 생성 실패: {pdf_err}")
            
            if st.session_state.temp_ref_docs:
                with st.expander("🔍 클라우드 인용 대조 (RAG 출처)"):
                    for idx, doc in enumerate(st.session_state.temp_ref_docs):
                        st.markdown(f"**[{idx+1}] {os.path.basename(doc.metadata.get('source', '알수없음'))} (Page {doc.metadata.get('page', 0)+1})**")
                        st.caption(doc.page_content[:300] + "...")
        else:
            st.info("왼쪽 양식에 정보를 기입하고 계획서 단독 또는 계획서&보고서 일괄 생성 버튼을 클릭하면 결과가 이곳에 렌더링됩니다.")

# 탭 2: 수업 결과 보고 및 평가서 작성
with tab2:
    st.subheader("📊 수업 결과 및 학생 피드백 평가서")
    st.markdown("수업 후 관찰한 학생들의 강점, 실수 해결 과정 등을 정형화된 보고서와 교사 피드백 양식으로 컴파일합니다.")
    
    col1, col2 = st.columns([1, 1.3])
    with col1:
        rep_title = st.text_input("수업 주제 / 활동명", key="rep_title", placeholder="예: 엔트리를 활용한 5학년 정다각형 그리기 실습")
        implementations = st.text_area("주요 수업 진행 내용 및 활동 산출물", placeholder="예: 조별로 정삼각형부터 정육각형까지 그리는 반복 블록 알고리즘을 설계하고 패들렛에 캡처 화면 업로드", height=100)
        troubleshooting = st.text_area("관찰된 문제 상황 및 교사 지도 대책", placeholder="예: 정오각형 회전각(72도)을 계산할 때 외각의 개념을 몰라 108도로 넣어 오작동 발생 $\rightarrow$ 칠판에 분필로 삼각형, 오각형 외각 회전 기하 시각화 설명 후 해결하도록 조치", height=100)
        outcomes = st.text_area("수업 성과 및 교사 성찰", placeholder="예: 조원들이 서로 머리를 맞대고 각도를 추론하여 협동적 문제 해결 역량이 신장됨. 시간이 모자라 정리 퀴즈가 생략된 점은 아쉬움", height=80)
        rep_add_reqs = st.text_area("보고서 추가 옵션", key="rep_add_reqs", placeholder="예: 교육청 제출용 격식 있는 공문 양식 표를 앞부분에 추가해줘", height=60)
        
        generate_report_btn = st.button("📊 상세 수업 평가서 생성", type="primary", use_container_width=True)
        
    with col2:
        st.markdown("#### 📄 결과 보고서 출력 및 RAG 검증")
        if generate_report_btn:
            if not rep_title or not implementations:
                st.error("수업 주제와 활동 내용을 기입해 주세요.")
            else:
                # 🚨 대기열 검문소 가동 (동시 생성 시 1순위 독점 보장)
                status = queue_manager.get_queue_status(session_id)
                if not status['is_my_turn']:
                    st.error(f"🚨 아직 회원님의 차례가 아닙니다. 현재 `{status['active_user_name']}`님이 AI를 점유 중입니다. 회원님은 `{status['my_turn']}번째` 대기 중입니다.")
                else:
                    try:
                        with st.spinner("AI 장학 비서가 수업 평가 보고서를 작성하고 있습니다..."):
                            report_result, ref_docs = st.session_state.rag_manager.generate_report(
                                project_title=rep_title,
                                implementations=implementations,
                                troubleshooting=troubleshooting,
                                outcomes=outcomes,
                                additional_requirements=rep_add_reqs
                            )
                            st.session_state.temp_report = report_result
                            st.session_state.temp_report_ref_docs = ref_docs
                    except Exception as e:
                        st.error(f"작성 실패: {e}")
                    finally:
                        # 🚨 연산 완료/예외 발생 무관하게 즉시 대기열 자격을 밀어주고 뒤 차례 대기자로 넘김
                        queue_manager.release_turn(session_id)
                        st.rerun()
                        
        # 세션 기반으로 보고서 내용과 다운로드 버튼 안전 렌더링 (Rerun 후 유실 차단)
        if st.session_state.temp_report:
            ref_docs = st.session_state.temp_report_ref_docs
            if ref_docs and any(doc.metadata.get('source') for doc in ref_docs):
                st.info("🔍 **[구글 클라우드 RAG 모드 검증]** 분석에 교육과정 표준이 매핑되었습니다.")
            else:
                st.info("💡 **[Gemma-4 단독 모드]** 모델 내부의 초등 학업 성취 평가 노하우를 바탕으로 결과를 종합했습니다.")
                
            st.markdown(st.session_state.temp_report)
            
            col_dl1, col_dl2 = st.columns([1, 1])
            with col_dl1:
                st.download_button(
                    label="📥 결과 보고서(.md) 다운로드",
                    data=st.session_state.temp_report,
                    file_name=f"수업결과보고_{rep_title.replace(' ', '_')}.md" if rep_title else "수업결과보고.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col_dl2:
                try:
                    pdf_data = markdown_to_pdf_bytes(st.session_state.temp_report, title=f"수업결과보고: {rep_title}" if rep_title else "수업결과보고")
                    st.download_button(
                        label="📥 PDF 파일(.pdf) 다운로드",
                        data=pdf_data,
                        file_name=f"수업결과보고_{rep_title.replace(' ', '_')}.pdf" if rep_title else "수업결과보고.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as pdf_err:
                    st.warning(f"PDF 생성 실패: {pdf_err}")
            
            if ref_docs:
                with st.expander("🔍 클라우드 인용 대조 (RAG 출처)"):
                    for idx, doc in enumerate(ref_docs):
                        st.markdown(f"**[{idx+1}] {os.path.basename(doc.metadata.get('source', '알수없음'))} (Page {doc.metadata.get('page', 0)+1})**")
                        st.caption(doc.page_content[:300] + "...")
        else:
            st.info("왼쪽 수업 실적란을 채워 제출하면, 학무 양식에 맞춘 풍성한 결과 평가서가 이곳에 채워집니다.")

# 탭 3: RAG 문서 챗봇
with tab3:
    st.subheader("💬 초등 교육과정 및 수업 팁 Q&A 비서")
    st.markdown("클라우드에서 동기화된 문서나 AI 교육 비서에게 수업 고민, 지도 팁, 교과 가이드라인을 바로 질문해보세요.")
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "ref_docs" in message and message["ref_docs"]:
                with st.expander("🔍 클라우드 인용 출처"):
                    for idx, doc in enumerate(message["ref_docs"]):
                        st.markdown(f"**[{idx+1}] {os.path.basename(doc.metadata.get('source', '알수없음'))} (Page {doc.metadata.get('page', 0)+1})**")
                        st.caption(doc.page_content[:200] + "...")

    if prompt := st.chat_input("질문을 입력해보세요. (예: 5학년 실과 교과서에서 '선택 구조' 성취기준을 어떻게 해석해야 하나요?)"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("답변을 마련하는 중..."):
                try:
                    ans, ref_docs = st.session_state.rag_manager.answer_question(prompt)
                    st.markdown(ans)
                    
                    if ref_docs:
                        with st.expander("🔍 클라우드 인용 출처"):
                            for idx, doc in enumerate(ref_docs):
                                st.markdown(f"**[{idx+1}] {os.path.basename(doc.metadata.get('source', '알수없음'))} (Page {doc.metadata.get('page', 0)+1})**")
                                st.caption(doc.page_content[:200] + "...")
                                
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": ans,
                        "ref_docs": ref_docs
                    })
                except Exception as e:
                    st.error(f"질의응답 오류: {e}")

# 탭 4: GPU 가속 인덱서 (관리자 전용)
if st.session_state.is_admin:
    with tab4:
        st.subheader("⚡ 대용량 PDF-to-FAISS GPU 가속 인덱서")
        st.markdown("외솔.한국 RAG 홈페이지 백엔드 탑재 및 대량의 교육과정 참고 문서를 고속 임베딩하기 위한 가속 모듈입니다.")
        
        col_system, col_settings = st.columns([1, 1.2])
        
        with col_system:
            st.markdown("### 🖥️ 시스템 인프라 현황")
            if cuda_available:
                st.success(f"🟢 **GPU 가속(CUDA) 활성화됨**\n- 디바이스명: `{gpu_name}`")
                try:
                    allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 2)
                    cached_mem = torch.cuda.memory_reserved(0) / (1024 ** 2)
                    st.info(f"💾 **GPU 메모리 상태**\n- 할당된 메모리: `{allocated_mem:.1f} MB`\n- 캐시된 메모리: `{cached_mem:.1f} MB`")
                except Exception:
                    pass
            else:
                st.warning("🟡 **CPU 연산 모드 (GPU 사용 불가)**\n- GPU 가속이 비활성화 상태입니다. 대용량 문서 인덱싱 속도가 다소 느려질 수 있습니다.")
                st.info("""
                💡 **GPU 가속(CUDA)을 활성화하려면:**
                1. NVIDIA 그래픽 드라이버 설치
                2. 로컬 가상 환경에 CUDA 버전의 PyTorch 재설치:
                ```bash
                pip uninstall torch torchvision -y
                pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
                ```
                """)
                
            st.markdown("---")
            st.markdown("### ☁️ 클라우드 연동 상태 (GCS / GDrive)")
            has_gcs = bool(GCS_BUCKET_NAME)
            has_gdrive = bool(GD_FOLDER_ID)
            
            if has_gcs:
                st.success(f"✅ **GCS 버킷 설정됨**: `{GCS_BUCKET_NAME}`")
            if has_gdrive:
                st.success(f"✅ **구글 드라이브 폴더 설정됨**: `{GD_FOLDER_ID[:12]}...`")
                
            if GOOGLE_APPLICATION_CREDENTIALS:
                st.caption(f"자격증명 경로: `{os.path.basename(GOOGLE_APPLICATION_CREDENTIALS)}`")
                
            if not has_gcs and not has_gdrive:
                st.warning("⚠️ **연동된 원격 저장소 없음**: 빌드 후 로컬 파일 저장만 수행 가능합니다.")
                
        with col_settings:
            st.markdown("### ⚙️ 인덱싱 구성 설정")
            
            idx_input_path = st.text_input("📁 PDF 입력 디렉토리 경로 (학습 데이터 폴더)", value=DATA_DIR, key="idx_input_path")
            idx_output_path = st.text_input("📁 FAISS 출력 디렉토리 경로 (벡터스토어 저장소)", value=VECTOR_DB_DIR, key="idx_output_path")
            
            idx_model_name = st.text_input("🏷️ 임베딩 모델 (HuggingFace)", value="jhgan/ko-sroberta-multitask", key="idx_model_name")
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                idx_chunk_size = st.number_input("청크 크기 (글자 수)", min_value=100, max_value=2000, value=600, step=50, key="idx_chunk_size")
                idx_overlap = st.number_input("청크 오버랩 (글자 수)", min_value=0, max_value=1000, value=100, step=10, key="idx_overlap")
            with col_c2:
                idx_batch_size = st.number_input("배치 처리 파일 수 (OOM 방지)", min_value=1, max_value=200, value=20, step=5, key="idx_batch_size")
                idx_upload_gcs = st.checkbox("GCS 버킷에 자동 업로드", value=has_gcs, disabled=not has_gcs, key="idx_upload_gcs")
                idx_upload_gdrive = st.checkbox("구글 드라이브 폴더에 자동 업로드", value=has_gdrive, disabled=not has_gdrive, key="idx_upload_gdrive")
                
            run_idx_btn = st.button("🚀 GPU 가속 인덱싱 시작", type="primary", use_container_width=True, key="run_idx_btn")

        st.markdown("---")
        st.markdown("### 📊 인덱싱 진행 과정 및 로깅")
        
        idx_log_area = st.empty()
        idx_progress_bar = st.progress(0.0)
        idx_status_text = st.empty()
        
        if run_idx_btn:
            idx_logs = []
            idx_log_placeholder = st.empty()
            
            def app_gui_callback(msg_type, data):
                if msg_type == "log":
                    idx_logs.append(data)
                    idx_log_placeholder.code("\n".join(idx_logs[-15:]))
                elif msg_type == "progress":
                    idx_progress_bar.progress(data["percent"])
                    idx_status_text.write(data["text"])
                    
            try:
                idx_status_text.write("⏳ 인덱싱 초기화 진행 중...")
                success, files, chunks = run_indexing(
                    input_dir=idx_input_path,
                    output_dir=idx_output_path,
                    model_name=idx_model_name,
                    file_batch_size=idx_batch_size,
                    text_chunk_size=idx_chunk_size,
                    text_chunk_overlap=idx_overlap,
                    upload_gcs=idx_upload_gcs,
                    upload_gdrive=idx_upload_gdrive,
                    progress_callback=app_gui_callback
                )
                
                if success:
                    st.success(f"🎉 **인덱싱 완료!** 총 {files}개 파일에서 {chunks}개의 벡터 청크를 성공적으로 가속 처리하여 저장 완료했습니다.")
                    if idx_upload_gcs:
                        st.info("☁️ 빌드된 FAISS 데이터베이스가 구글 클라우드 스토리지(GCS)에 성공적으로 반영되었습니다.")
                    if idx_upload_gdrive:
                        st.info("☁️ 빌드된 FAISS 데이터베이스가 구글 드라이브(Google Drive) 공유 폴더에 성공적으로 반영되었습니다.")
                    
                    # 인덱서가 완료되면 현재 앱의 RAG 엔진에 사용 중인 벡터 스토어도 새로 고침
                    if "vector_manager" in st.session_state and "rag_manager" in st.session_state:
                        try:
                            vs = st.session_state.vector_manager.load_vector_store()
                            if vs:
                                st.session_state.rag_manager.set_vector_store(vs)
                                st.info("🔄 **현재 앱 세션의 AI 엔진 참조 데이터베이스 동기화 완료!**")
                        except Exception as reload_err:
                            st.warning(f"참조 DB 자동 동기화 중 경고: {reload_err}")
                else:
                    st.error("❌ 인덱싱 작업 중 문제 또는 데이터 없음으로 인해 중단되었습니다. 로그를 확인하십시오.")
            except Exception as e:
                st.error(f"💥 인덱싱 중 런타임 예외가 발생했습니다: {e}")
                import traceback
                st.code(traceback.format_exc())
