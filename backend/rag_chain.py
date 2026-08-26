import os
import requests
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

class RAGChainManager:
    def __init__(self, ollama_host, model_name, vector_store=None):
        # 🚨 OLLAMA 주소가 localhost인 경우 윈도우 IPv6 소켓 오류(WinError 10049) 방지를 위해 127.0.0.1로 강제 보정
        if "localhost" in ollama_host:
            ollama_host = ollama_host.replace("localhost", "127.0.0.1")
            
        # 1. 교사님의 로컬 PC Ollama 원격 개방 연결 상태 확인
        local_connected = False
        target_ollama_host = ollama_host
        
        # 🚨 로컬 윈도우 환경(os.name == 'nt')인 경우에는 불필요한 외부 터널(localtunnel.me) 루프백을 생략하고 127.0.0.1로 즉시 직결
        is_windows = (os.name == 'nt')
        
        if config.MY_LOCAL_OLLAMA_URL and not is_windows:
            try:
                ping_url = f"{config.MY_LOCAL_OLLAMA_URL.rstrip('/')}/api/tags"
                res = requests.get(ping_url, timeout=2.0)
                if res.status_code == 200:
                    local_connected = True
                    target_ollama_host = MY_LOCAL_OLLAMA_URL
                    print(f"Successfully connected to Remote Local PC Ollama at: {MY_LOCAL_OLLAMA_URL}")
            except Exception as e:
                print(f"Remote Local PC Ollama is offline: {e}")
                
        # 2. 엔진 인스턴스 분기 수립 (컨텍스트 윈도우 num_ctx 및 generation_config 2중 안전 장치로 출력 짤림 완벽 방어)
        if local_connected:
            self.llm = ChatOllama(
                base_url=target_ollama_host,
                model=model_name,
                temperature=0.4,
                num_predict=8192,
                num_ctx=16384
            )
            self.is_gemini_active = False
            self.connected_engine_info = "로컬 AI 컴퓨터 (gemma4)"
        elif config.GEMINI_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    api_key=config.GEMINI_API_KEY,
                    temperature=0.4
                )
                self.is_gemini_active = True
                self.connected_engine_info = "구글 Gemini 클라우드 엔진"
            except Exception as e:
                print(f"Error initializing Gemini: {e}")
                self.llm = ChatOllama(
                    base_url=ollama_host,
                    model=model_name,
                    temperature=0.4,
                    num_predict=8192,
                    num_ctx=16384
                )
                self.is_gemini_active = False
                self.connected_engine_info = "기본 로컬 Ollama"
        else:
            self.llm = ChatOllama(
                base_url=ollama_host,
                model=model_name,
                temperature=0.4,
                num_predict=8192,
                num_ctx=16384
            )
            self.is_gemini_active = False
            self.connected_engine_info = "기본 로컬 Ollama"
        self.vector_store = vector_store
        self.retriever = None
        if self.vector_store:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 7}
            )

    def set_vector_store(self, vector_store):
        self.vector_store = vector_store
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 7}
        )

    def _format_docs(self, docs):
        if not docs:
            return "제공된 외부 참고 문서가 없습니다. 본인이 학습한 배경 지식에 의존하여 풍부하게 응답하십시오."
        formatted = []
        for doc in docs:
            source = os.path.basename(doc.metadata.get('source', '알수없음'))
            page = doc.metadata.get('page', 0) + 1
            formatted.append(f"[출처: {source} (Page {page})]\n{doc.page_content}")
        return "\n\n".join(formatted)

    def generate_study_plan(self, project_title, learning_goals, duration, level, model_type, additional_requirements):
        # 1. 문서 검색 (RAG)
        query = f"{project_title} {learning_goals} {model_type} {additional_requirements}"
        docs = []
        if self.retriever:
            try:
                docs = self.retriever.invoke(query)
            except Exception as e:
                print(f"Error retrieving documents: {e}")
        context = self._format_docs(docs)
        
        # 2. 상세 프롬프트 정의
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", """당신은 초등 교육학 및 교육과정 설계 전문가입니다. 
초등학교 교사가 교실에서 학생들과 바로 활용할 수 있는 매우 체계적이고 세부적인 **'초등 프로젝트 학습 상세 계획서'**를 작성해야 합니다.
 
[중요 지침]
1. 제공된 참고 문서(Context)에 성취기준, 평가 기준, 교사용 지침서, 실습 방법 등의 구체적인 교육 활동 가이드라인이 적혀 있다면, AI의 기존 일반 상식보다 **이 가이드라인을 최우선 가치로 삼아 계획서 전반에 100% 강제 반영하여 실습을 기획**하십시오.
2. 특히 참고 문서(Context)에 명시된 고유의 독창적 활동(예: 특정 퀴즈 진행, 특정 물품 조립 등)이 감지되면, 이를 무조건 지도안의 도입/전개/정리 시나리오 내에 누락 없이 삽입하여 시나리오를 구성해 주어야 합니다.
3. **일반 교과 코딩 편향 금지**: 정보/실과 과목이 아니거나 사용자가 코딩/엔트리/컴퓨터 실습을 직접 요구하지 않은 경우에는 **컴퓨터 코딩, 알고리즘, 엔트리 블록 코딩, 소스코드 등 IT/SW 관련 기술 내용을 절대 독단적으로 추가하지 마십시오.** 대신 초등학생 눈높이에 맞춘 일반적인 아동 중심 활동(모형 만들기, 역할극, 모둠 토의, 야외 조사, 캠페인 포스터 그리기 등)으로 교수학습 시나리오를 가득 채워 주십시오.
4. 만약 참고 문서(Context)가 비어 있거나 관련 정보가 적다면, 당신이 내장한 교육학적 지식을 100% 발휘하여 풍부하고 상세한 계획서를 작성해야 합니다. 절대 요약하거나 축소하지 마십시오.
5. 각 주차(또는 차시)의 교수학습 설계 계획은 **반드시 다음 7가지 마크다운 헤더를 엄격하게 준수하여 순서대로 상세히 기술**해야 하며, 피로도 경감 등을 핑계로 임의로 목차를 누락시키지 마십시오:
   - **### [차시/주차 번호] 수업 주제**
   - **교육과정 성취기준 연계**: 해당 차시 수업과 연계되는 **초등 국가 교육과정 공식 성취기준 코드 및 성취기준명(예: '[6과05-02] 온도 변화에 따른 기체의 부피 변화를 관찰하고...와 연계')**을 제공된 참고 문서(Context) 내의 실제 교육과정 데이터에서 정확히 추출하여 무조건 1줄로 표기하십시오. (가상의 임의 코드를 만들어내지 말고, RAG 데이터 내의 성취기준과 수업 내용을 정교하게 1:1 대조하여 매핑할 것)
   - **학습 주제 & 핵심 질문**: 수업의 본질적 목적을 이끌어내는 핵심 질문
   - **교사의 역할 및 발문**: 교사가 각 단계에서 학생들에게 던질 수 있는 실제 발문 예시 및 수업 지침
   - **학생의 활동**: 학생들이 수업 중에 모둠별 또는 개별로 수행할 구체적인 행동 및 토론, 실험 양상
   - **지도상 유의점 & 예외 상황**: 초등학생 수준에서 발생 가능한 기기 파손, 통제 불능, 안전사고에 대한 주의사항
   - **수행 평가 루브릭 (Rubric)**: 해당 차시 활동을 관찰 평가할 수 있도록 '평가 요소', '상 (우수)', '중 (보통)', '하 (노력 요함)'로 구성된 **4열 마크다운 표(Table)**를 작성하십시오. 성취 등급 기준은 초등학생의 관찰 가능한 구체적 행동 기술어(예: "~할 수 있다")로 차등 묘사해야 합니다.
6. 가독성을 위해 마크다운 테이블, 구분선, 이모지 등을 적극 활용하십시오.
7. **절대 중간 생략 및 요약 꼼수 금지**: 출력량 한계를 핑계로 중간에 작성을 끊거나, "(이후 차시는 발표 준비로 구성됩니다)" 등의 괄호 요약 구문으로 설계를 때우며 도망치는 게으른 행위를 엄격히 금지합니다. 지정된 차시 분량(예: 8차시면 8차시 전부)에 해당하는 모든 차시를 1차시부터 마지막 차시까지 8가지 구조적 마크다운 헤더를 단 한 개도 건너뛰지 않고 100% 완전하게 끝까지 서술하여 출력을 마쳐야 합니다.
8. **핵심 위주의 실무적 서술 (출력 다이어트)**: 8차시 전체의 방대한 교육 설계 내용이 잘림 없이 한 번에 완결되도록, 교사 발문과 학생 활동 및 루브릭 서술 시 미사여구와 불필요한 반복 구문을 철저히 배제하고 **실무에 꼭 필요한 개조식 및 요약식 핵심 문장 위주로 컴팩트하게 서술**하십시오. 텍스트의 불필요한 거품을 줄임으로써 전체 8차시 분량이 물리적인 토큰 잘림 현상 없이 100% 한 벌의 완성본으로 최종 인쇄될 수 있게 전체 출력 볼륨을 지능적으로 분배하여 조율하십시오.
"""),
            ("human", """[사용자 요청 정보]
- 프로젝트/수업 주제: {project_title}
- 핵심 학습 목표: {learning_goals}
- 수업 대상 (난이도): {level}
- 진행 기간: {duration}
- 적용할 수업 모형: {model_type}
- 참고 문서 내용 (RAG):
{context}
- 교사 추가 희망 사항: {additional_requirements}

위 요건들을 바탕으로 초등학교 수업에 즉시 투입 가능한 매우 상세하고 구체적인 프로젝트 학습 계획서를 완성해 주세요.""")
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

    def generate_report(self, project_title, implementations, troubleshooting, outcomes, additional_requirements):
        # 1. 문서 검색 (RAG)
        query = f"{project_title} {implementations} {troubleshooting}"
        docs = []
        if self.retriever:
            try:
                docs = self.retriever.invoke(query)
            except Exception as e:
                print(f"Error retrieving documents for report: {e}")
        context = self._format_docs(docs)
        
        # 2. 상세 프롬프트 정의
        prompt_tmpl = ChatPromptTemplate.from_messages([
            ("system", """당신은 초등학교 교육 평가 전문가이자 장학사입니다. 
교사가 프로젝트 수업을 수행한 후, 교육청이나 학교 내부 보고용으로 제출할 수 있는 격식 있고 상세한 **'프로젝트 수업 결과 및 평가 보고서'**를 작성해 주세요.

[중요 지침]
1. GCS 참고 문서(Context)가 있다면 교육과정 성취기준 대조 및 평가 지표 정합성을 검증하여 기술하십시오.
2. 참고 문서가 없더라도, 당신이 가진 초등 교육 평가 역량을 총동원하여 매우 풍부하고 전문적인 결과 보고서를 생성해 주어야 합니다.
3. 보고서에는 반드시 다음 구성이 상세히 포함되어야 합니다:
   - **수업 개요 및 추진 목표**
   - **수업 활동별 세부 구현 양상** (학생들이 산출한 결과물의 구체적 묘사)
   - **수업 중 발생한 문제 상황 및 교사의 교육적 조치 (수업 내 Troubleshooting)**: 학생들이 어려워했던 부분과 교사가 피드백을 통해 극복한 과정 기술
   - **학생의 역량 변화 성과 및 평가 결과**: 초등 교육 관점에서의 핵심 역량(협력, 창의성, 문제해결 등) 변화 요약
   - **향후 수업 개선을 위한 제언 및 후속 학습 처방**
4. 전문 장학 자료처럼 정형화되고 일목요연한 마크다운 문서로 포맷팅해 주세요.
"""),
            ("human", """[사용자 요청 정보]
- 수업/프로젝트 주제: {project_title}
- 주요 수업 활동 및 산출물 내용: {implementations}
- 학생들의 한계점 및 교사의 피드백 해결책: {troubleshooting}
- 관찰된 성과 및 배운 점: {outcomes}
- 참고 문서 내용 (RAG):
{context}
- 교사 추가 요구사항: {additional_requirements}

위 요건들을 바탕으로 신뢰성 높고 상세한 프로젝트 수업 결과 보고서를 작성해 주세요.""")
        ])
        
        chain = prompt_tmpl | self.llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "project_title": project_title,
            "implementations": implementations,
            "troubleshooting": troubleshooting,
            "outcomes": outcomes,
            "additional_requirements": additional_requirements
        })
        return response, docs

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

[참고 문서 내용 (RAG)]
{context}

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
   - **학생의 역량 성과 및 가상의 평가 결과 표**: 계획서의 수행평가 루브릭 기준에 준하여, 가상의 학생 성취도 분포(상/중/하 명수) 및 구체적인 학생별 관찰 평어 기록 예시(루브릭에 근거한 행동 특성 묘사)를 담은 테이블 제공
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
