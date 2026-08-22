import streamlit as st
import os
import json
import requests
from config import OLLAMA_HOST, OLLAMA_MODEL, DATA_DIR, VECTOR_DB_DIR, GCS_BUCKET_NAME, GOOGLE_APPLICATION_CREDENTIALS
from backend.gcs_manager import GCSManager
from backend.vector_store import VectorStoreManager
from backend.rag_chain import RAGChainManager
from backend.pdf_generator import markdown_to_pdf_bytes

st.set_page_config(
    page_title="초등 프로젝트 수업 계획 및 평가 비서",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 임베딩 모델 캐싱
@st.cache_resource
def load_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'}
    )

# 세션 상태 초기화
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

if "vector_manager" not in st.session_state:
    with st.spinner("교사용 AI 엔진 초기화 중... (최초 실행 시 다소 시간이 소요됩니다)"):
        # Windows 한글 경로 우회를 위한 로컬 기본 가이드 문서 자동 복사
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
    res = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
    if res.status_code == 200:
        ollama_connected = True
except:
    pass

# 사이드바
with st.sidebar:
    st.markdown("### 🏫 교육 자료 저장소 설정")
    
    gcs_active = st.session_state.gcs_manager and st.session_state.gcs_manager.is_connected()
    
    if gcs_active:
        st.success(f"☁️ **구글 클라우드 교재 연동 완료**\n- 버킷: `{st.session_state.gcs_manager.bucket_name}`")
    else:
        st.info("💻 **로컬 기본 자료 모드**\n- 내장 교육과정 문서로 구동됩니다.")
        
    with st.expander("⚙️ 구글 클라우드 상세 연결 설정"):
        st.caption("학교 전용 클라우드 스토리지를 연결하려면 자격증명 파일(.json)과 버킷명을 지정해 주세요.")
        uploaded_key = st.file_uploader("GCP 서비스 계정 JSON 키 파일 업로드", type=["json"])
        manual_bucket = st.text_input("GCS 버킷명 입력", value=GCS_BUCKET_NAME if GCS_BUCKET_NAME else "")
        if uploaded_key and manual_bucket:
            try:
                key_data = json.load(uploaded_key)
                st.session_state.gcs_manager = GCSManager(
                    bucket_name=manual_bucket,
                    credentials_info=key_data
                )
                st.success("수동 자격 증명으로 연결되었습니다!")
                st.rerun()
            except Exception as e:
                st.error(f"인증 파일 오류: {e}")
                
    st.markdown("---")
    st.markdown("### 📥 교육 자료 동기화")
    st.caption("클라우드에 새로 수집한 교과서나 학습 지도서 문서를 가져와 인공지능에 학습시킵니다.")
    
    if st.button("✨ 내 구글 클라우드 자료 동기화 및 학습", use_container_width=True, type="primary"):
        if not st.session_state.gcs_manager or not st.session_state.gcs_manager.is_connected():
            st.warning("연결된 구글 클라우드가 없습니다. 로컬 기본 자료를 바탕으로 인덱싱을 수행합니다.")
            with st.spinner("로컬 교과 가이드북 학습 중..."):
                vs = st.session_state.vector_manager.build_vector_store()
                if vs:
                    st.session_state.rag_manager.set_vector_store(vs)
                    st.success("로컬 참고자료 동기화 완료!")
        else:
            with st.spinner("구글 클라우드에서 최신 자료 가져오는 중..."):
                downloaded = st.session_state.gcs_manager.sync_all_files(DATA_DIR)
                st.success(f"클라우드로부터 {len(downloaded)}개 문서 다운로드 완료!")
            with st.spinner("자료 분석 및 벡터 DB화 진행 중..."):
                vs = st.session_state.vector_manager.build_vector_store()
                if vs:
                    st.session_state.rag_manager.set_vector_store(vs)
                    st.success("구글 클라우드 문서 학습 완료!")

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

# 탭 구성
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
        outcomes = st.text_area("수업 성과 및 교사 성찰", placeholder="예: 조원들이 서로 머리를 맞대고 각도를 추론하여 협동적 문제 해결 역량이 신장됨. 시간이 모자라 정리 퀴즈가 생략된 점은 아쉬움", height=80)
        rep_add_reqs = st.text_area("보고서 추가 옵션", key="rep_add_reqs", placeholder="예: 교육청 제출용 격식 있는 공문 양식 표를 앞부분에 추가해줘", height=60)
        
        generate_report_btn = st.button("📊 상세 수업 평가서 생성", type="primary", use_container_width=True)
        
    with col2:
        st.markdown("#### 📄 결과 보고서 출력 및 RAG 검증")
        if generate_report_btn:
            if not rep_title or not implementations:
                st.error("수업 주제와 활동 내용을 기입해 주세요.")
            elif not ollama_connected:
                st.error("Ollama(gemma4) 서비스 연동을 확인해 주세요.")
            else:
                with st.spinner("AI 장학 비서가 수업 평가 보고서를 작성하고 있습니다..."):
                    try:
                        report_result, ref_docs = st.session_state.rag_manager.generate_report(
                            project_title=rep_title,
                            implementations=implementations,
                            troubleshooting=troubleshooting,
                            outcomes=outcomes,
                            additional_requirements=rep_add_reqs
                        )
                        
                        if ref_docs and any(doc.metadata.get('source') for doc in ref_docs):
                            st.info("🔍 **[구글 클라우드 RAG 모드 검증]** 분석에 교육과정 표준이 매핑되었습니다.")
                        else:
                            st.info("💡 **[Gemma-4 단독 모드]** 모델 내부의 초등 학업 성취 평가 노하우를 바탕으로 결과를 종합했습니다.")
                            
                        st.markdown(report_result)
                        
                        col_dl1, col_dl2 = st.columns([1, 1])
                        with col_dl1:
                            st.download_button(
                                label="📥 결과 보고서(.md) 다운로드",
                                data=report_result,
                                file_name=f"수업결과보고_{rep_title.replace(' ', '_')}.md",
                                mime="text/markdown",
                                use_container_width=True
                            )
                        with col_dl2:
                            try:
                                pdf_data = markdown_to_pdf_bytes(report_result, title=f"수업결과보고: {rep_title}")
                                st.download_button(
                                    label="📥 PDF 파일(.pdf) 다운로드",
                                    data=pdf_data,
                                    file_name=f"수업결과보고_{rep_title.replace(' ', '_')}.pdf",
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
                    except Exception as e:
                        st.error(f"작성 실패: {e}")
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
        
        if not ollama_connected:
            st.error("Ollama 서비스 미구동 상태입니다.")
        else:
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
