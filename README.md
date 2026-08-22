# 🏫 초등 교사용 RAG 기반 수업 계획서 & 결과 보고서 일괄 생성기

> **초등학교 교사들의 수업 기획 및 문서 업무 부담을 혁신적으로 경감하기 위한 RAG 기반 스마트 교육 행정 지원 서비스입니다.**

본 프로젝트는 초등 교육과정에 특화된 로컬 교육 가이드 문서 또는 사용자의 **구글 클라우드 스토리지(GCS)** 내 보관 문서들을 RAG(검색 증강 생성) 기술로 자동 분석하여, 고상세도의 수업 계획서(지도안)와 가상의 사후 수업 결과 보고서를 단 한 번의 클릭으로 동시 생성해 줍니다.

---

## ✨ 핵심 기능

1. **📅 상세 수업 계획서 & 지도안 설계**
   - 초등 1~6학년 군별 특성에 맞춘 수업 모델 제공 (PBL, 놀이 중심, 실험 실습, 토의 토론 등).
   - 수업 목표, 성취기준 반영 근거, 주차별/차시별 세부 활동(교사 발문, 학생 활동, 지도상 유의점 포함)을 고밀도로 생성.

2. **📊 가상 수업 결과 보고서 시뮬레이션 (일괄 생성)**
   - 생성된 계획서를 자동으로 이어받아, 실제 수업을 진행했을 때 발생 가능한 **가상의 결과 보고서**와 **학생 평가서**를 일괄 빌드.
   - 수업 중 학생들이 겪기 쉬운 오개념(예: 각도 계산 실수, 맞춤법 등)과 이를 극복하게 해 준 **교사의 현장 피드백/해결 지도 시나리오(Troubleshooting)**를 자동 포함하여 장학 및 제출 양식 수준의 결과물 도출.

3. **📥 고품질 한글 PDF / 마크다운 파일 다운로드**
   - FPDF2 라이브러리와 Windows 맑은 고딕(`malgun.ttf`) 폰트 간의 한글 행간 붕괴 및 겹침 현상을 해결하기 위해 **절대 좌표 수동 Y축 갱신 드로잉 기법** 도입.
   - 폰트 미지원 특수 기호를 이모지 정화 필터(`clean_emojis`)로 걸러내고 표 데이터를 **카드 요약 단락(Card Section)**으로 정돈하여, 깨짐·잘림·겹침이 없는 완벽한 한글 PDF 출력 보장.

4. **☁️ GCS & 로컬 가이드 하이브리드 RAG 파이프라인**
   - 구글 클라우드 스토리지 버킷 연동을 지원하여, 교사가 수집한 구글 클라우드의 고유 교육 자료를 우선 검색에 활용.
   - GCS 연동 부재 시, 기본 탑재된 `초등_교육과정_가이드.txt`를 로컬 임베딩하여 RAG 기반 검증을 상시 수행.

5. **💬 초등 교육과정 전문 Q&A 챗봇**
   - 수업 구상 중 궁금한 성취기준이나 교수설계 방법론에 대해 현직 교사 관점에서 전문적이고 따뜻하게 답변해 주는 실무 Q&A 어시스턴트 기능.

---

## 🛠️ 기술 스택 및 라이브러리

- **Frontend / UI**: `Streamlit` (고급스러운 다크 블루 교사용 커스텀 CSS 테마 탑재)
- **AI Core**: `Ollama (Gemma-4 로컬 탑재 모델)` / `LangChain`
- **Vector DB**: `FAISS` / `HuggingFaceEmbeddings`
- **GCS Integration**: `google-cloud-storage`
- **PDF Engine**: `fpdf2` (수동 절대 좌표 드로잉 커스터마이징)

---

## 🚀 시작하기 (설치 및 실행 방법)

### 1. 필수 선행 설정 (Ollama)
로컬 환경에 [Ollama](https://ollama.com/)를 설치한 뒤, 사용 중인 Gemma 모델을 다운로드하여 실행해 둡니다.
```bash
ollama pull gemma
```

### 2. 환경 변수 설정 (`.env`)
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 아래 양식에 맞추어 작성합니다.
```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma

# 구글 클라우드 스토리지 RAG 검증용 (옵션)
GCS_BUCKET_NAME=your-gcs-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=your-gcs-service-account-key.json
```

### 3. 패키지 설치 및 실행
```bash
# 의존성 패키지 설치
pip install -r requirements.txt

# Streamlit 앱 가동
streamlit run app.py
```
실행이 완료되면 브라우저에서 **`http://localhost:8501`**로 자동 연결됩니다.

---

## 📂 프로젝트 폴더 구조

```text
├── backend/
│   ├── gcs_sync.py        # Google Cloud Storage 파일 동기화 모듈
│   ├── vector_store.py    # FAISS 벡터스토어 인덱스 빌드 및 관리
│   ├── rag_chain.py       # 주차 계획서/시뮬레이션 보고서 프롬프트 및 RAG 체인 설계
│   └── pdf_generator.py   # 수동 Y축 6.2mm 이동 절대 좌표 한글 PDF 생성 모듈
├── data/
│   └── 초등_교육과정_가이드.txt  # 기본 내장 로컬 RAG용 학습 가이드
├── .env                   # 환경설정 파일 (Ollama 주소 및 GCS 연동 정보)
├── config.py              # Windows 한글 경로 에러 우회용 영문 공용 디렉토리 지정
├── app.py                 # Streamlit 통합 다크 블루 테마 대시보드 메인
└── README.md              # 프로젝트 매뉴얼 (본 문서)
```

---

## 💡 Windows 환경 경로 격리 가이드 (FAISS)
Windows 환경의 사용자 계정명에 한글이나 공백이 포함된 경우, FAISS C++ 내부의 파일 쓰기 API가 파일 경로를 정상적으로 저장하지 못하고 `Illegal byte sequence` 쓰기 에러를 발생시킵니다.
이를 우회하기 위해 본 솔루션은 운영체제 권한이 확실하게 보장되고 100% 영문으로만 이루어진 공용 데이터 디렉토리인 **`C:/Users/Public/.elementary_assistant`** 폴더를 내부 캐시 및 임베딩 저장 공간으로 지정하여 경로 충돌 문제를 완전히 해결했습니다.
