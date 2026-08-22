# 📦 초등 수업 계획 및 결과 보고 일괄 생성기 - 통합 소스 코드 번들

본 문서는 프로젝트의 모든 핵심 소스 코드와 설정 파일을 하나로 묶은 통합 문서입니다.  
이 문서를 그대로 **ChatGPT(GPT-4o)** 또는 **Gemini 1.5 Pro/Advanced** 등에 복사하여 업로드하면, 전체 아키텍처를 분석받고 교육적 관점이나 기능상의 개선 피드백을 매우 손쉽게 구하실 수 있습니다.

---

## 📂 프로젝트 파일 구조
- `config.py` : Windows 사용자명 한글 경로 우회 및 기본 설정
- `app.py` : Streamlit 다크 블루 교사용 통합 대시보드
- `backend/pdf_generator.py` : 수동 Y축 6.2mm 이동 절대 좌표 한글 PDF 생성 모듈 (이모지 정화 필터 포함)
- `backend/rag_chain.py` : RAG 체인 설계, 수업계획 및 수업 결과 보고 일괄 생성 프롬프트 체인
- `backend/vector_store.py` : FAISS 로컬 벡터 인덱스 로드 및 빌드 매니저
- `backend/gcs_manager.py` : Google Cloud Storage 연동 및 파일 다운로드/동기화 헬퍼
- `.env` (템플릿) : 설정용 환경변수 파일

---

## 1. config.py
```python
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

# --- 경로 설정 변경 (한글 사용자명으로 인한 FAISS Illegal byte sequence 완벽 회피) ---
# Windows 사용자 이름에 한글(예: 박세훈)이 포함된 경우, 홈 디렉토리 경로에도 한글이 들어가 에러가 납니다.
# 따라서 100% 영문으로만 이루어지고 쓰기 권한이 자유로운 공용 폴더(C:/Users/Public)를 임시/영구 데이터 저장소로 사용합니다.
BASE_DATA_DIR = "C:/Users/Public/.elementary_assistant"

DATA_DIR = os.path.join(BASE_DATA_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DATA_DIR, "vector_store")

# 폴더 생성 보장
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
```

---

## 2. app.py
```python
import os
import streamlit as st
import requests
from config import OLLAMA_HOST, OLLAMA_MODEL, DATA_DIR, VECTOR_DB_DIR, GCS_BUCKET_NAME, GOOGLE_APPLICATION_CREDENTIALS
from backend.gcs_manager import GCSManager
from backend.vector_store import VectorStoreManager
from backend.rag_chain import RAGChainManager
from backend.pdf_generator import markdown_to_pdf_bytes

st.set_page_config(
    page_title="초등 프로젝트 학습 계획서 및 보고서 생성기",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'}
    )

# 1. GCS 및 RAG 매니저 지연(Lazy) 초기화
if "gcs_manager" not in st.session_state:
    if GCS_BUCKET_NAME:
        creds_path = GOOGLE_APPLICATION_CREDENTIALS
        if creds_path and not os.path.isabs(creds_path):
            creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), creds_path)
            
        st.session_state.gcs_manager = GCSManager(
            bucket_name=GCS_BUCKET_NAME,
            credentials_path=creds_path if os.path.exists(creds_path) else None
        )
    else:
        st.session_state.gcs_manager = None

if "vector_manager" not in st.session_state:
    with st.spinner("교사용 AI 엔진 초기화 중... (최초 실행 시 다소 시간이 소요됩니다)"):
        import shutil
        src_guide = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "초등_교육과정_가이드.txt")
        dst_guide = os.path.join(DATA_DIR, "초등_교육과정_가이드.txt")
        if os.path.exists(src_guide) and not os.path.exists(dst_guide):
            shutil.copy(src_guide, dst_guide)
            
        embeddings = load_embeddings()
        st.session_state.vector_manager = VectorStoreManager(DATA_DIR, VECTOR_DB_DIR, embeddings=embeddings)
        vs = st.session_state.vector_manager.load_vector_store()
        if not vs and os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
            vs = st.session_state.vector_manager.build_vector_store()
        st.session_state.rag_manager = RAGChainManager(OLLAMA_HOST, OLLAMA_MODEL, vs)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "temp_plan" not in st.session_state:
    st.session_state.temp_plan = ""
if "temp_sim_report" not in st.session_state:
    st.session_state.temp_sim_report = ""
if "temp_ref_docs" not in st.session_state:
    st.session_state.temp_ref_docs = []

# 고급스러운 다크/블루 톤 테마 CSS
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
        color: #f1f5f9;
    }
    .sidebar .sidebar-content {
        background: #1e293b;
    }
    div.stButton > button:first-child {
        background-color: #0284c7;
        color: white;
        border-radius: 6px;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #0369a1;
        transform: translateY(-1px);
    }
    .stTextInput>div>div>input {
        background-color: #1e293b;
        color: #f1f5f9;
        border: 1px solid #475569;
    }
    .stTextArea>div>div>textarea {
        background-color: #1e293b;
        color: #f1f5f9;
        border: 1px solid #475569;
    }
    .stSelectbox>div>div {
        background-color: #1e293b;
        color: #f1f5f9;
    }
</style>
""", unsafe_allow_html=True)

# Ollama 상태 검사
ollama_connected = False
try:
    resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
    if resp.status_code == 200:
        ollama_connected = True
except:
    pass

st.title("🏫 초등 프로젝트 학습(PBL) 계획서 및 수업 결과 보고서 생성기")
st.markdown("현직 초등학교 교사의 교육과정 재구성 및 수업 행정 문서를 효율화하기 위한 RAG 기반 AI 어시스턴트입니다.")

# 사이드바
with st.sidebar:
    st.header("⚙️ 교육 데이터 동기화")
    if ollama_connected:
        st.success(f"🟢 LLM 연동 성공 ({OLLAMA_MODEL})")
    else:
        st.error("🔴 Ollama 연결 실패 (로컬 서비스 미구동)")
        
    st.markdown("---")
    st.subheader("☁️ 구글 클라우드 스토리지 (GCS)")
    if st.session_state.gcs_manager and st.session_state.gcs_manager.is_connected():
        st.success(f"GCS 연결 완료: {GCS_BUCKET_NAME}")
        
        if st.button("🔄 클라우드 문서 동기화 및 학습"):
            with st.spinner("GCS 파일 다운로드 및 벡터 인덱스 갱신 중..."):
                try:
                    downloaded = st.session_state.gcs_manager.sync_all_files(DATA_DIR)
                    if downloaded:
                        st.session_state.vector_manager.build_vector_store()
                        st.success(f"성공: {len(downloaded)}개 문서 동기화 및 벡터 DB 구축 완료!")
                        st.rerun()
                    else:
                        st.info("클라우드 버킷에 동기화할 새 문서가 없습니다.")
                except Exception as e:
                    st.error(f"동기화 에러: {e}")
    else:
        st.warning("GCS 연동이 설정되지 않았습니다. (.env에 버킷명과 JSON 키 이름을 지정하면 연동됩니다)")
        
    st.markdown("---")
    st.subheader("📂 로컬 저장소 정보")
    st.text(f"저장 경로:\n{DATA_DIR}")
    try:
        local_files = os.listdir(DATA_DIR)
        st.caption(f"현재 보관 문서: {len(local_files)}개")
        for f in local_files[:5]:
            st.caption(f"- {f}")
        if len(local_files) > 5:
            st.caption("...")
    except:
        st.caption("문서 없음")

# 탭 분리
tab1, tab2, tab3 = st.tabs([
    "📅 상세 수업 계획 및 지도안 설계", 
    "📊 수업 결과 보고 및 평가서 작성", 
    "💬 초등 교육과정 실무 Q&A"
])

# 탭 1: 상세 수업 계획 설계
with tab1:
    st.subheader("📅 초등 주차별 상세 프로젝트 계획서 및 지도안")
    st.markdown("프로젝트 수업 주제와 학습 목표에 기반하여 성취기준 중심의 상세 수업 지도안을 뼈대부터 일차별/차시별 계획까지 빌드합니다.")
    
    col1, col2 = st.columns([1, 1.3])
    with col1:
        proj_title = st.text_input("수업 프로젝트 주제", placeholder="예: 우리 동네 쓰레기 배출 문제 해결하기")
        learning_goals = st.text_area("핵심 학습 목표", placeholder="예: 생활 주변의 쓰레기 분리배출 실태를 조사하고, 자원 순환의 필요성을 깨달아 실천 방안을 제시할 수 있다.", height=80)
        
        c_dur, c_lvl = st.columns(2)
        with c_dur:
            duration = st.selectbox("수업 분량 (차시/기간)", ["단기(1~2차시)", "중기(4~8차시)", "장기(10차시 이상)"])
        with c_lvl:
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
            elif not ollama_connected:
                st.error("Ollama(gemma4) 서비스가 작동 중인지 확인해 주세요.")
            else:
                # 1단계: 계획서 생성
                with st.spinner("AI 엔진이 상세 수업계획서를 집필 중입니다..."):
                    try:
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
                    except Exception as e:
                        st.error(f"계획서 생성 실패: {e}")
                        
                # 2단계: 일괄 생성 시 결과 보고서도 연속 생성
                if generate_both_btn and st.session_state.temp_plan:
                    with st.spinner("이 계획에 기반한 가상의 결과 보고서를 시뮬레이션하여 작성 중입니다..."):
                        try:
                            sim_report = st.session_state.rag_manager.generate_simulated_report(
                                project_title=proj_title,
                                learning_goals=learning_goals,
                                plan_content=st.session_state.temp_plan,
                                level=level,
                                model_type=model_type
                            )
                            st.session_state.temp_sim_report = sim_report
                        except Exception as e:
                            st.error(f"가상 보고서 생성 실패: {e}")

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
        strong_points = st.text_area("아동들의 주요 성과 및 관찰된 역량 변화", placeholder="예: 외각 원리를 깨달은 후 조원들끼리 서로 디버깅을 도와주며 문제해결력과 정보적 효능감이 크게 향상됨", height=80)
        generate_report_btn = st.button("🚀 정밀 결과 보고서 생성", type="primary", use_container_width=True)
        
    with col2:
        st.markdown("#### 📄 결과 보고서 및 학생 평가 양식")
        if generate_report_btn:
            if not rep_title or not implementations:
                st.error("수업 주제와 주요 진행 내용을 반드시 기입해 주세요.")
            elif not ollama_connected:
                st.error("Ollama(gemma4) 서비스가 작동 중인지 확인해 주세요.")
            else:
                with st.spinner("AI 엔진이 결과 평가서를 컴파일 중입니다..."):
                    try:
                        report_res = st.session_state.rag_manager.generate_result_report(
                            title=rep_title,
                            implementations=implementations,
                            troubleshooting=troubleshooting,
                            strong_points=strong_points
                        )
                        st.markdown(report_res)
                        
                        col_dl3, col_dl4 = st.columns([1, 1])
                        with col_dl3:
                            st.download_button(
                                label="📥 마크다운 파일(.md) 다운로드",
                                data=report_res,
                                file_name=f"수업결과보고서_{rep_title.replace(' ', '_')}.md",
                                mime="text/markdown",
                                use_container_width=True
                            )
                        with col_dl4:
                            try:
                                pdf_data = markdown_to_pdf_bytes(report_res, title=f"수업결과보고서: {rep_title}")
                                st.download_button(
                                    label="📥 PDF 파일(.pdf) 다운로드",
                                    data=pdf_data,
                                    file_name=f"수업결과보고서_{rep_title.replace(' ', '_')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                            except Exception as pdf_err:
                                st.warning(f"PDF 생성 실패: {pdf_err}")
                    except Exception as e:
                        st.error(f"생성 실패: {e}")
        else:
            st.info("왼쪽 평가 양식에 관찰 일지를 입력하고 생성 버튼을 누르면 정형화된 보고용 문서가 이곳에 출력됩니다.")

# 탭 3: 초등 교육과정 실무 Q&A
with tab3:
    st.subheader("💬 초등 교육과정 및 수업설계 현장 상담소")
    st.markdown("수업을 설계하거나 성취기준을 분석할 때 발생하는 실무적인 의문점에 대해 편하게 질문해 주십시오.")
    
    # 대화 기록 표시
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # 사용자 질문 입력
    if user_query := st.chat_input("질문 내용을 입력하세요 (예: 6학년 소수의 나눗셈 지도 시 시각적 모델 활용 방법은?)"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant"):
            with st.spinner("교육과정 문서를 토대로 답변을 준비하고 있습니다..."):
                try:
                    ans, docs = st.session_state.rag_manager.answer_question(user_query)
                    st.markdown(ans)
                    
                    if docs:
                        with st.expander("📚 답변 구성에 참고한 문서"):
                            for idx, doc in enumerate(docs):
                                st.markdown(f"**[{idx+1}] {os.path.basename(doc.metadata.get('source', '가이드라인'))}**")
                                st.caption(doc.page_content[:200] + "...")
                                
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                except Exception as e:
                    st.error(f"대화 처리 중 오류가 발생했습니다: {e}")
```

---

## 3. backend/pdf_generator.py
```python
import os
from fpdf import FPDF

class UnicodePDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        try:
            self.set_font("Malgun", size=9)
            self.cell(0, 10, f"페이지 {self.page_no()} / {{nb}}", align="C")
        except:
            self.set_font("Helvetica", size=9)
            self.cell(0, 10, f"Page {self.page_no()} / {{nb}}", align="C")

def clean_emojis(text):
    """맑은 고딕 폰트에서 미지원하는 이모지 및 특수 기호를 텍스트 설명으로 치환하거나 정화합니다."""
    if not text:
        return ""
    replacements = {
        "💡": "[팁] ",
        "🔍": "[출처] ",
        "🏫": "[초등] ",
        "📝": "[계획] ",
        "🚀": "[시작] ",
        "📊": "[보고] ",
        "💬": "[대화] ",
        "✨": "[안내] ",
        "🟢": "● ",
        "🔴": "○ ",
        "⚪": "○ ",
        "📥": "[다운로드] ",
        "▪": "▪ ",
        "•": "• ",
        "🗓️": "[일정] ",
        "🎯": "[목표] ",
        "✅": "[완료] "
    }
    for emoji, repl in replacements.items():
        text = text.replace(emoji, repl)
    
    return text

def wrap_korean_text(text, limit=48):
    """한글 단어 잘림 및 우측 마진 침범 방지를 위해 지정 글자수 단위로 강제 줄바꿈을 수행합니다."""
    if not text:
        return ""
    
    lines = text.split('\n')
    wrapped_lines = []
    
    for line in lines:
        if len(line) <= limit:
            wrapped_lines.append(line)
            continue
            
        current_line = ""
        current_len = 0
        
        words = line.split(' ')
        for word in words:
            if len(word) > limit:
                for char in word:
                    if current_len + 1 > limit:
                        wrapped_lines.append(current_line)
                        current_line = char
                        current_len = 1
                    else:
                        current_line += char
                        current_len += 1
            elif current_len + len(word) + (1 if current_len > 0 else 0) > limit:
                wrapped_lines.append(current_line)
                current_line = word
                current_len = len(word)
            else:
                if current_len > 0:
                    current_line += " " + word
                    current_len += 1 + len(word)
                else:
                    current_line = word
                    current_len = len(word)
                    
        if current_line:
            wrapped_lines.append(current_line)
            
    return '\n'.join(wrapped_lines)

def markdown_to_pdf_bytes(md_text, title="수업계획서"):
    pdf = UnicodePDF()
    pdf.alias_nb_pages()
    
    # 기본 여백 선언
    pdf.set_margins(15, 20, 15)
    pdf.add_page()
    
    font_path = "C:\\Windows\\Fonts\\malgun.ttf"
    font_bold_path = "C:\\Windows\\Fonts\\malgunbd.ttf"
    
    has_font = False
    if os.path.exists(font_path):
        try:
            pdf.add_font("Malgun", "", font_path)
            if os.path.exists(font_bold_path):
                pdf.add_font("Malgun", "B", font_bold_path)
            has_font = True
        except Exception as e:
            print(f"Font load error: {e}")
            
    # --- Y축 좌표 수동 통제 및 강제 개행 절대 좌표 렌더링 ---
    # fpdf2 한글 폰트 행간 붕괴 버그(겹침 현상)를 완벽히 격파합니다.
    current_y = 30
    left_margin = 15
    line_height = 6.2  # 줄 간격 고정 수치
    
    def write_safe_line(txt, size=10, is_bold=False):
        nonlocal current_y
        
        # 하단 도달 시 페이지 넘김 및 Y축 리셋
        if current_y > 270:
            pdf.add_page()
            current_y = 25
            
        if has_font:
            pdf.set_font("Malgun", "B" if is_bold else "", size=size)
        else:
            pdf.set_font("Helvetica", "B" if is_bold else "", size=size)
            
        txt = clean_emojis(txt)
        
        # text() 함수를 사용하여 X좌표 l_margin(15mm)에 강제 고정 렌더링
        pdf.text(left_margin, current_y + (size * 0.22), txt)
        current_y += line_height

    # 1. 제목 출력 (중앙 정렬 좌표 동적 계산)
    clean_title = clean_emojis(title)
    if has_font:
        pdf.set_font("Malgun", "B", size=16)
    else:
        pdf.set_font("Helvetica", "B", size=16)
        
    title_lines = wrap_korean_text(clean_title, limit=35).split('\n')
    for tl in title_lines:
        str_w = pdf.get_string_width(tl)
        center_x = left_margin + (pdf.epw - str_w) / 2
        pdf.text(center_x, current_y + 4, tl)
        current_y += 10
        
    current_y += 4
    
    # 2. 본문 한 줄씩 파싱하여 절대 좌표 렌더링
    lines = md_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        line = line.replace('\u2013', '-').replace('\u2014', '-').replace('\u2022', '*').replace('\u2019', "'").replace('\u2018', "'").replace('\u201d', '"').replace('\u201c', '"')
        line = clean_emojis(line)
        
        # --- 표(Table)를 카드 요약 단락으로 변환하여 순차 Y축 출력 ---
        if line.strip().startswith('|'):
            if '-|-' in line or '---|---' in line or ':---' in line:
                i += 1
                continue
                
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if not parts or all(p == '' for p in parts):
                i += 1
                continue
                
            header_lbl = parts[0].replace('**', '').replace('*', '').strip()
            write_safe_line(f"▪ {header_lbl} :", size=10, is_bold=True)
            
            desc_val = " | ".join([p for p in parts[1:] if p]).replace('**', '').replace('*', '').strip()
            if desc_val:
                wrapped_desc = wrap_korean_text(desc_val, limit=48)
                for dl in wrapped_desc.split('\n'):
                    write_safe_line(f"   {dl}", size=9.5, is_bold=False)
                    
            current_y += 1.5
            i += 1
            continue
            
        # 일반 라인 처리
        if line.startswith('# '):
            current_y += 4
            write_safe_line(line[2:], size=13.5, is_bold=True)
            current_y += 2
        elif line.startswith('## '):
            current_y += 3
            write_safe_line(line[3:], size=11.5, is_bold=True)
            current_y += 1.5
        elif line.startswith('### ') or line.startswith('#### '):
            current_y += 2
            prefix_len = 4 if line.startswith('### ') else 5
            write_safe_line(line[prefix_len:], size=10, is_bold=True)
            current_y += 1
        else:
            clean_line = line.replace('**', '').replace('*', '').strip()
            if not clean_line:
                current_y += 3
                i += 1
                continue
                
            wrapped_line = wrap_korean_text(clean_line, limit=52)
            for wl in wrapped_line.split('\n'):
                write_safe_line(wl, size=10, is_bold=False)
                
        i += 1
        
    return bytes(pdf.output())
```

---

## 4. backend/rag_chain.py
```python
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGChainManager:
    def __init__(self, host, model, vector_store=None):
        self.llm = Ollama(base_url=host, model=model)
        self.vector_store = vector_store
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3}) if vector_store else None

    def _format_docs(self, docs):
        if not docs:
            return "참조할 문서 내용이 존재하지 않습니다."
        return "\n\n".join([f"[출처: {d.metadata.get('source', '가이드')}] {d.page_content}" for d in docs])

    def generate_study_plan(self, project_title, learning_goals, duration, level, model_type, additional_requirements):
        # 1. 문서 검색 (RAG)
        query = f"{project_title} {learning_goals} {model_type} 초등 지도안"
        docs = []
        if self.retriever:
            try:
                docs = self.retriever.invoke(query)
            except Exception as e:
                print(f"Error retrieving documents: {e}")
        context = self._format_docs(docs)

        # 2. 프롬프트 정의
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", """당신은 초등학교 교수설계 전문가이자 수석 교사입니다. 
제시된 수업 정보와 참고 문서 내용(RAG)을 기반으로 초등학교 수업에 즉시 활용할 수 있는 매우 상세한 **'프로젝트 학습 상세 계획서 및 지도안'**을 작성해 주세요.

[중요 지침]
1. 타겟 학년의 발달 특성에 맞는 어조와 어휘를 선택해 주십시오. (예: 초등 1~2학년은 구체적 조작과 놀이 중심, 5~6학년은 자기주도적 PBL 및 메타인지 강조)
2. 성취기준과 연계된 타당한 평가 계획을 함께 명시하십시오.
3. 교수학습 흐름(시나리오)은 교사의 실제 발문과 학생들의 구체적인 활동 양상, 그리고 지도상 유의점이 반드시 각 주차/차시별로 모두 기술되도록 아주 구체적으로 만들어 주십시오.
4. 마크다운 형식으로 작성해 주십시오.
"""),
            ("human", """[수업 정보]
- 프로젝트 주제: {project_title}
- 핵심 학습 목표: {learning_goals}
- 수업 분량: {duration}
- 대상 학년: {level}
- 교수학습 모형: {model_type}
- 참고 문서 내용 (RAG):
{context}

[교사의 추가 요구사항]
{additional_requirements}

위 지침과 입력을 반영하여 교실에 바로 적용할 수 있는 상세 수업계획서를 완성해 주세요.""")
        ])

        chain = prompt_tmpl | self.llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "project_title": project_title,
            "learning_goals": learning_goals,
            "duration": duration,
            "level": level,
            "model_type": model_type,
            "additional_requirements": additional_requirements
        })
        return response, docs

    def generate_result_report(self, title, implementations, troubleshooting, strong_points):
        # 1. 문서 검색 (RAG)
        query = f"{title} 수업결과보고서 성과 평가서"
        docs = []
        if self.retriever:
            try:
                docs = self.retriever.invoke(query)
            except Exception as e:
                print(f"Error retrieving documents for report: {e}")
        context = self._format_docs(docs)

        # 2. 프롬프트 정의
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", """당신은 장학 활동 및 교육 성과 평가를 전담하는 초등 수석 교사입니다.
교사가 기입한 수업 관찰 사실을 바탕으로 교육청 및 학교 내부 결재용 **'수업 결과 및 학생 역량 피드백 보고서'**를 정형화된 격식체로 컴파일해 주십시오.

[중요 지침]
1. 수업 진행 실적과 교육적 변화를 공문서 형식으로 구조화하여 정갈하게 서술하십시오.
2. 성취기준 달성도 관점에서의 피드백과 향후 개선을 위한 제언을 상세하게 포함해야 합니다.
3. 마크다운 형식으로 작성해 주십시오.
"""),
            ("human", """[교사의 수업 관찰 일지]
- 활동명: {title}
- 실적 및 산출물: {implementations}
- 문제 상황과 지도 대책 (Troubleshooting): {troubleshooting}
- 아동의 핵심 성과/역량 변화: {strong_points}
- 참고 가이드 문서 (RAG):
{context}

위 내용을 완성도 높은 결재용 보고서로 렌더링해 주세요.""")
        ])

        chain = prompt_tmpl | self.llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "title": title,
            "implementations": implementations,
            "troubleshooting": troubleshooting,
            "strong_points": strong_points
        })
        return response

    def answer_question(self, question):
        # 1. 문서 검색 (RAG)
        docs = []
        if self.retriever:
            try:
                docs = self.retriever.invoke(question)
            except Exception as e:
                print(f"Error retrieving documents for QA: {e}")
        context = self._format_docs(docs)
        
        # 2. 프롬프트 정의
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", """당신은 초등 교사를 돕는 전문적인 교육과정 Q&A 인공지능 비서입니다.
초등 교육과정, 교수학습 설계, 교실 운영, 학생 지도 등 교사의 전문적인 질문에 성실하게 답변해 주십시오.

[지침]
1. 제공된 참고 문서(Context)에 관련 내용이 있다면 이를 명시하고 핵심 요약을 제시하십시오.
2. 참고 문서에 명시되지 않은 질문이더라도 교사의 실무에 실질적인 도움이 되는 교육적 방법론과 지식을 바탕으로 친절하고 상세하게 답변하십시오.
"""),
            ("human", "{question}")
        ])
        
        chain = prompt_tmpl | self.llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "question": question
        })
        return response, docs

    def generate_simulated_report(self, project_title, learning_goals, plan_content, level, model_type):
        # 1. 문서 검색 (RAG)
        query = f"{project_title} {learning_goals} {model_type} 평가서 결과 보고서"
        docs = []
        if self.retriever:
            try:
                docs = self.retriever.invoke(query)
            except Exception as e:
                print(f"Error retrieving documents for simulated report: {e}")
        context = self._format_docs(docs)
        
        # 2. 프롬프트 정의
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", """당신은 초등학교 교육 평가 전문가이자 장학사입니다. 
제공된 '수업 계획서(Plan)'를 바탕으로, 해당 계획서대로 실제 초등학교 교실에서 수업이 진행되었다고 가정하고,
그 결과물로서 제출할 수 있는 격식 있고 상세한 **'가상의 프로젝트 수업 결과 및 학생 평가 보고서'**를 작성해 주세요.

[중요 지침]
1. 계획서(Plan)에 기재된 학습 활동 흐름과 학년 수준을 충실히 반영하여 시뮬레이션 하십시오.
2. 결과 보고서에는 반드시 다음 사항이 포함되어야 합니다:
   - **가상의 주요 수업 진행 실적**: 계획대로 진행되었을 때 학생들이 도출한 구체적인 결과물 예시 묘사
   - **실제 관찰되었을 법한 학생들의 한계 및 교사 조치 (Troubleshooting 시뮬레이션)**: 해당 학년 아동들이 이 실습이나 활동 중 겪기 쉬운 오개념(예: 각도 계산 실수, 맞춤법 등)과 교사가 현장에서 어떻게 개별 피드백을 주어 해결했는지 구체적 시나리오 제시
   - **학생의 역량 성과 및 가상의 평가 결과 표**
   - **차시 개선을 위한 교사의 제언**
3. 마크다운 테이블과 깔끔한 구조로 작성해 주세요.
"""),
            ("human", """[수업 정보]
- 수업 주제: {project_title}
- 대상 학년: {level}
- 수업 모형: {model_type}
- 참고 문서 내용 (RAG):
{context}

[작성된 수업 계획서]
{plan_content}

위 계획서에 기반하여, 가상의 실제 수업 결과 보고서 및 평가서를 상세하게 완성해 주세요.""")
        ])
        
        chain = prompt_tmpl | self.llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "project_title": project_title,
            "level": level,
            "model_type": model_type,
            "plan_content": plan_content
        })
        return response
```

---

## 5. backend/vector_store.py
```python
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    def __init__(self, data_dir, vector_db_dir, embeddings=None):
        self.data_dir = data_dir
        self.vector_db_dir = vector_db_dir
        self.embeddings = embeddings

    def get_embeddings(self):
        """임베딩 모델을 필요할 때 지연 로딩(Lazy Loading)합니다."""
        if self.embeddings is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': 'cpu'}
            )
        return self.embeddings

    def load_vector_store(self):
        """저장된 FAISS 벡터 인덱스를 로드합니다."""
        if os.path.exists(os.path.join(self.vector_db_dir, "index.faiss")):
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_db_dir, 
                    self.get_embeddings(),
                    allow_dangerous_deserialization=True
                )
                return self.vector_store
            except Exception as e:
                print(f"Error loading vector store: {e}")
        return None

    def build_vector_store(self):
        """data_dir에 있는 모든 문서(pdf, txt 등)를 읽어서 벡터 인덱스를 생성합니다."""
        documents = []
        
        if not os.path.exists(self.data_dir):
            return None
            
        for file in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file)
            if os.path.isdir(file_path):
                continue
                
            try:
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading file {file}: {e}")

        if not documents:
            print("No documents found to index.")
            return None

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)

        # FAISS 빌드 및 로컬 저장
        self.vector_store = FAISS.from_documents(splits, self.get_embeddings())
        self.vector_store.save_local(self.vector_db_dir)
        return self.vector_store
        
    def add_single_document(self, file_path):
        """새로운 문서 1개를 벡터 스토어에 점진적으로 추가합니다."""
        documents = []
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading file for increment: {e}")
            return None
            
        if not documents:
            return None
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(documents)
        
        self.load_vector_store()
        if self.vector_store:
            self.vector_store.add_documents(splits)
            self.vector_store.save_local(self.vector_db_dir)
        else:
            self.vector_store = FAISS.from_documents(splits, self.get_embeddings())
            self.vector_store.save_local(self.vector_db_dir)
            
        return self.vector_store
```

---

## 6. backend/gcs_manager.py
```python
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
```

---

## 7. .env (예시 설정 템플릿)
```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma

# Google Cloud Storage 연동 (선택 사항)
GCS_BUCKET_NAME=your_gcs_bucket_name
GOOGLE_APPLICATION_CREDENTIALS=your_google_key_filename.json
```
