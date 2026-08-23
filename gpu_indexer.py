import os
import sys
import gc
import time
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# 🚨 인코딩 에러 원천 차단 (Windows 콘솔 환경 대응)
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
        sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
    except Exception:
        pass

# Streamlit 구동 여부 확인
try:
    import streamlit as st
    is_streamlit_run = st.runtime.exists()
except ImportError:
    is_streamlit_run = False

# PyTorch 및 CUDA 확인
import torch
cuda_available = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"

# LangChain 관련 모듈 임포트
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# 프로젝트 설정 및 모듈 임포트 시도
try:
    from config import DATA_DIR, VECTOR_DB_DIR, GCS_BUCKET_NAME, GOOGLE_APPLICATION_CREDENTIALS
    from backend.gcs_manager import GCSManager
except ImportError:
    # 폴더 외부에서 단독 실행 시 기본 경로 설정
    DATA_DIR = "./data"
    VECTOR_DB_DIR = "./vector_store"
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    GCSManager = None

# 병렬 PDF 텍스트 추출 헬퍼 함수 가져오기
from backend.pdf_extractor import extract_pdf_pages


def run_indexing(input_dir, output_dir, model_name, file_batch_size, text_chunk_size, text_chunk_overlap, upload_gcs=False, progress_callback=None):
    """
    대용량 PDF 문서를 GPU 가속을 활용하여 인덱싱하는 코어 함수입니다.
    """
    start_time = time.time()
    
    if not os.path.exists(input_dir):
        raise ValueError(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 대상 PDF 파일 목록화
    pdf_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(input_dir, f))
    ]
    
    total_files = len(pdf_files)
    if total_files == 0:
        if progress_callback:
            progress_callback("log", "인덱싱할 PDF 파일이 입력 폴더에 없습니다.")
        return False, 0, 0
        
    if progress_callback:
        progress_callback("log", f"총 {total_files}개의 PDF 문서를 감지했습니다. 인덱싱을 시작합니다.")
        
    # 2. 임베딩 모델 로드 (GPU 가속 설정)
    device = "cuda" if cuda_available else "cpu"
    if progress_callback:
        progress_callback("log", f"임베딩 모델 로드 중 ({model_name} on {device.upper()})...")
        
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'batch_size': 128 if device == "cuda" else 32}  # GPU 사용 시 대용량 배치를 적용하여 속도 향상
    )
    
    # 텍스트 스플리터
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=text_chunk_size,
        chunk_overlap=text_chunk_overlap
    )
    
    main_index = None
    processed_count = 0
    total_chunks = 0
    
    # 멀티프로세싱 코어 개수 설정 (CPU 물리코어 수에 맞춰 튜닝)
    cpu_cores = max(1, multiprocessing.cpu_count() - 1)
    
    # OOM 방지를 위해 PDF 파일들을 배치 단위로 나누어 인덱싱 및 점진적 병합 수행
    for i in range(0, total_files, file_batch_size):
        batch_files = pdf_files[i:i + file_batch_size]
        batch_num = (i // file_batch_size) + 1
        total_batches = (total_files + file_batch_size - 1) // file_batch_size
        
        if progress_callback:
            progress_callback("log", f"\n[배치 {batch_num}/{total_batches}] {len(batch_files)}개 PDF 병렬 분석 시작...")
            
        # 3. CPU 병렬 처리를 통해 텍스트 추출 (CPU 바운드 작업 최적화)
        extracted_pages = []
        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            future_to_file = {executor.submit(extract_pdf_pages, f): f for f in batch_files}
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    pages = future.result()
                    if pages:
                        extracted_pages.extend(pages)
                except Exception as exc:
                    if progress_callback:
                        progress_callback("log", f"파일 파싱 중 에러 발생: {os.path.basename(file_path)} - {exc}")
                        
        if not extracted_pages:
            if progress_callback:
                progress_callback("log", f"[배치 {batch_num}] 추출된 텍스트가 없습니다. 건너뜁니다.")
            continue
            
        # 4. LangChain Document 객체로 변환
        langchain_docs = [
            Document(page_content=page["text"], metadata=page["metadata"])
            for page in extracted_pages
        ]
        
        # 5. 텍스트 분할 (Chunking)
        splits = text_splitter.split_documents(langchain_docs)
        total_chunks += len(splits)
        processed_count += len(batch_files)
        
        if progress_callback:
            progress_callback("log", f"[배치 {batch_num}] {len(extracted_pages)}개 페이지를 {len(splits)}개 텍스트 청크로 분할했습니다. GPU 임베딩 생성 시작...")
            if progress_callback:
                progress_callback("progress", {
                    "percent": min(1.0, processed_count / total_files),
                    "text": f"인덱싱 진행률: {processed_count}/{total_files}개 완료 ({len(splits)}개 청크 임베딩 중...)"
                })
                
        # 6. FAISS 인덱스 빌드 및 병합
        temp_index = FAISS.from_documents(splits, embeddings)
        
        if main_index is None:
            main_index = temp_index
        else:
            main_index.merge_from(temp_index)
            
        # 7. VRAM 및 시스템 메모리 관리 (가비지 컬렉터 강제 호출 및 PyTorch 캐시 비우기)
        del temp_index
        del langchain_docs
        del splits
        gc.collect()
        if cuda_available:
            torch.cuda.empty_cache()
            
    # 8. 로컬 파일 저장
    if main_index:
        if progress_callback:
            progress_callback("log", f"로컬 디렉토리에 FAISS 벡터스토어 저장 중: {output_dir}")
        main_index.save_local(output_dir)
    else:
        if progress_callback:
            progress_callback("log", "인덱싱 결과가 비어 있어 벡터스토어를 저장하지 못했습니다.")
        return False, 0, 0
        
    duration = time.time() - start_time
    if progress_callback:
        progress_callback("log", f"\n✨ [로컬 빌드 완료] 처리시간: {duration:.2f}초 | 총 파일: {total_files}개 | 생성된 청크: {total_chunks}개")
        
    # 9. 구글 클라우드 스토리지(GCS) 연동 및 업로드
    if upload_gcs:
        if GCSManager and GCS_BUCKET_NAME:
            # GCP 자격 증명 확인
            creds_path = GOOGLE_APPLICATION_CREDENTIALS
            if creds_path and not os.path.isabs(creds_path):
                # 윈도우/리눅스 절대 경로 대응
                creds_path = os.path.abspath(os.path.join(os.path.dirname(__file__), creds_path))
                
            try:
                if progress_callback:
                    progress_callback("log", f"구글 클라우드 스토리지 업로드 시작... (버킷: {GCS_BUCKET_NAME})")
                    
                gcs = GCSManager(
                    bucket_name=GCS_BUCKET_NAME,
                    credentials_path=creds_path if os.path.exists(creds_path) else None
                )
                
                if gcs.is_connected():
                    # 로컬 인덱스 파일 업로드
                    faiss_file = os.path.join(output_dir, "index.faiss")
                    pkl_file = os.path.join(output_dir, "index.pkl")
                    
                    # RAG 홈페이지에서 인식할 수 있게 vector_store 폴더 경로로 업로드
                    gcs.upload_file(faiss_file, "vector_store/index.faiss")
                    gcs.upload_file(pkl_file, "vector_store/index.pkl")
                    
                    if progress_callback:
                        progress_callback("log", "✅ 구글 클라우드 스토리지로 FAISS 인덱스 동기화(업로드) 완료!")
                else:
                    if progress_callback:
                        progress_callback("log", "❌ GCS 연결에 실패했습니다. 자격 증명을 확인해 주세요.")
            except Exception as gcs_err:
                if progress_callback:
                    progress_callback("log", f"❌ GCS 업로드 중 에러 발생: {gcs_err}")
        else:
            if progress_callback:
                progress_callback("log", "⚠️ GCP 설정이 유효하지 않아 클라우드 업로드를 건너뜁니다. (config.py 또는 .env 확인)")
                
    return True, total_files, total_chunks


# --- CLI 모드 엔트리포인트 ---
def run_cli():
    parser = argparse.ArgumentParser(description="대용량 PDF-to-FAISS GPU 가속 인덱서 프로그램")
    parser.add_argument("--input-dir", type=str, default=DATA_DIR, help="PDF 파일들이 있는 입력 디렉토리 경로")
    parser.add_argument("--output-dir", type=str, default=VECTOR_DB_DIR, help="FAISS 인덱스를 저장할 출력 디렉토리 경로")
    parser.add_argument("--model-name", type=str, default="jhgan/ko-sroberta-multitask", help="임베딩 모델 이름")
    parser.add_argument("--batch-size", type=int, default=20, help="OOM 방지를 위한 배치당 PDF 파일 개수")
    parser.add_argument("--chunk-size", type=int, default=600, help="텍스트 청크 크기 (글자 수)")
    parser.add_argument("--overlap", type=int, default=100, help="텍스트 청크 오버랩 크기 (글자 수)")
    parser.add_argument("--upload-gcs", action="store_true", help="GCS 버킷에 빌드된 벡터스토어 업로드 여부")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 대용량 PDF-to-FAISS GPU 가속 인덱서 (CLI 모드)")
    print("=" * 60)
    print(f"- 입력 경로: {args.input_dir}")
    print(f"- 출력 경로: {args.output_dir}")
    print(f"- 임베딩 모델: {args.model_name}")
    print(f"- GPU 사용 가능 여부: {'🟢 사용 가능' if cuda_available else '🔴 사용 불가능 (CPU 모드로 구동)'}")
    if cuda_available:
        print(f"  └ 디바이스명: {gpu_name}")
    else:
        print("  💡 GPU 가속을 사용하려면 CUDA와 호환되는 PyTorch 설치가 필요합니다.")
        print("     설치 예시: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print(f"- GCS 자동 동기화 여부: {args.upload_gcs}")
    print("-" * 60)
    
    def cli_callback(msg_type, data):
        if msg_type == "log":
            print(data)
        elif msg_type == "progress":
            # 한 줄 출력을 통해 심플하게 진행률 묘사
            sys.stdout.write(f"\r[{data['text']}]")
            sys.stdout.flush()
            
    success, files, chunks = run_indexing(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        file_batch_size=args.batch_size,
        text_chunk_size=args.chunk_size,
        text_chunk_overlap=args.overlap,
        upload_gcs=args.upload_gcs,
        progress_callback=cli_callback
    )
    
    if success:
        print("\n\n✅ 인덱싱 작업이 안전하게 종료되었습니다.")
    else:
        print("\n\n❌ 인덱싱 작업이 실패하였습니다.")


# --- Streamlit GUI 모드 엔트리포인트 ---
def run_gui():
    st.set_page_config(
        page_title="대용량 PDF-to-FAISS GPU 가속 인덱서",
        page_icon="⚡",
        layout="wide"
    )
    
    # 다크 블루 고급스러운 스타일 지정
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #10B981, #3B82F6, #6366F1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        .desc-text {
            color: #94A3B8;
            font-size: 1.1rem;
            margin-bottom: 20px;
        }
        .gpu-card {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='main-title'>⚡ 대용량 PDF-to-FAISS GPU 가속 인덱서</div>", unsafe_allow_html=True)
    st.markdown("<div class='desc-text'>외솔.한국 RAG 홈페이지의 백엔드 성능을 강화하고 대량의 참고 교육자료를 초고속 임베딩하기 위한 가속 유틸리티입니다.</div>", unsafe_allow_html=True)
    
    col_system, col_settings = st.columns([1, 1.2])
    
    with col_system:
        st.subheader("🖥️ 시스템 인프라 현황")
        
        with st.container():
            if cuda_available:
                st.success(f"🟢 **GPU 가속(CUDA) 활성화됨**\n- 디바이스명: `{gpu_name}`")
                
                # GPU 상태 모니터링 시도
                try:
                    allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 2)
                    cached_mem = torch.cuda.memory_reserved(0) / (1024 ** 2)
                    st.info(f"💾 **GPU 메모리 상태**\n- 할당된 메모리: `{allocated_mem:.1f} MB`\n- 캐시된 메모리: `{cached_mem:.1f} MB`")
                except Exception:
                    pass
            else:
                st.warning("🟡 **CPU 연산 모드 (GPU 사용 불가)**\n- GPU 가속이 비활성화 상태입니다. 대용량 문서 인덱싱 속도가 다소 느려질 수 있습니다.")
                st.info("""
                💡 **GPU 가속(CUDA)을 사용하려면:**
                1. NVIDIA 그래픽 드라이버 설치
                2. 로컬 가상 환경에 CUDA 버전의 PyTorch 재설치:
                ```bash
                pip uninstall torch torchvision -y
                pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
                ```
                """)
                
        # 환경변수 요약
        st.markdown("---")
        st.markdown("### ☁️ 클라우드 연동 상태 (GCS)")
        has_gcs = bool(GCS_BUCKET_NAME)
        if has_gcs:
            st.success(f"✅ **GCS 버킷 설정됨**: `{GCS_BUCKET_NAME}`")
            if GOOGLE_APPLICATION_CREDENTIALS:
                st.caption(f"자격증명 경로: `{os.path.basename(GOOGLE_APPLICATION_CREDENTIALS)}`")
        else:
            st.warning("⚠️ **GCS 버킷 설정 없음**: 빌드 후 로컬 파일 저장만 수행 가능합니다.")
            
    with col_settings:
        st.subheader("⚙️ 인덱싱 구성 설정")
        
        input_path = st.text_input("📁 PDF 입력 디렉토리 경로 (학습 데이터 폴더)", value=DATA_DIR)
        output_path = st.text_input("📁 FAISS 출력 디렉토리 경로 (벡터스토어 저장소)", value=VECTOR_DB_DIR)
        
        model_name = st.text_input("🏷️ 임베딩 모델 (HuggingFace)", value="jhgan/ko-sroberta-multitask")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            chunk_size = st.number_input("청크 크기 (글자 수)", min_value=100, max_value=2000, value=600, step=50)
            overlap = st.number_input("청크 오버랩 (글자 수)", min_value=0, max_value=1000, value=100, step=10)
        with col_c2:
            batch_size = st.number_input("배치 처리 파일 수 (OOM 방지)", min_value=1, max_value=200, value=20, step=5)
            upload_gcs = st.checkbox("인덱싱 완료 후 구글 클라우드에 자동 업로드", value=has_gcs, disabled=not has_gcs)
            
        run_btn = st.button("🚀 GPU 가속 인덱싱 시작", type="primary", use_container_width=True)

    st.markdown("---")
    st.subheader("📊 인덱싱 진행 과정 및 로깅")
    
    log_area = st.empty()
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    if run_btn:
        logs = []
        log_placeholder = st.empty()
        
        def gui_callback(msg_type, data):
            if msg_type == "log":
                logs.append(data)
                # 역순으로 로그를 띄우거나 최신 로그를 계속 스크롤
                log_placeholder.code("\n".join(logs[-15:]))
            elif msg_type == "progress":
                progress_bar.progress(data["percent"])
                status_text.write(data["text"])
                
        try:
            status_text.write("⏳ 인덱싱 초기화 진행 중...")
            success, files, chunks = run_indexing(
                input_dir=input_path,
                output_dir=output_path,
                model_name=model_name,
                file_batch_size=batch_size,
                text_chunk_size=chunk_size,
                text_chunk_overlap=overlap,
                upload_gcs=upload_gcs,
                progress_callback=gui_callback
            )
            
            if success:
                st.success(f"🎉 **인덱싱 완료!** 총 {files}개 파일에서 {chunks}개의 벡터 청크를 성공적으로 가속 처리하여 저장 완료했습니다.")
                if upload_gcs:
                    st.info("☁️ 빌드된 FAISS 데이터베이스가 외솔.한국 RAG 홈페이지가 호스팅되는 구글 클라우드에 성공적으로 반영되었습니다.")
            else:
                st.error("❌ 인덱싱 작업 중 문제 또는 데이터 없음으로 인해 중단되었습니다. 로그를 확인하십시오.")
        except Exception as e:
            st.error(f"💥 인덱싱 중 런타임 예외가 발생했습니다: {e}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    # 프로세스 스폰 문제(Windows) 방지를 위해 메인 가드 지정
    multiprocessing.freeze_support()
    
    if is_streamlit_run:
        run_gui()
    else:
        run_cli()
