import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
import tempfile
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool

# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ✅ 요리 전문 웹 검색 툴 정의
def search_cooking_web():
    search = SerpAPIWrapper()
    
    def run_cooking_search(query: str) -> str:
        # 요리 관련 검색어 최적화
        cooking_query = f"{query} 레시피 요리법 만들기"
        results = search.results(cooking_query)
        organic = results.get("organic_results", [])
        formatted = []
        
        for r in organic[:5]:
            title = r.get("title")
            link = r.get("link")
            source = r.get("source")
            snippet = r.get("snippet")
            if link:
                formatted.append(f"🍽️ [{title}]({link}) (출처: {source})\n  📝 {snippet}")
            else:
                formatted.append(f"🍽️ {title} (출처: {source})\n  📝 {snippet}")
        
        if not formatted:
            return f"'{query}' 요리에 대한 정보를 찾지 못했습니다. 다른 요리명으로 다시 시도해보세요. 🤔"
        
        return "\n\n".join(formatted)
    
    return Tool(
        name="cooking_search",
        func=run_cooking_search,
        description="요리 레시피, 요리법, 재료 정보를 검색할 때 사용합니다. 특정 요리에 대한 자세한 정보를 제공합니다."
    )

# ✅ 요리책 PDF 업로드 → 벡터DB → 검색 툴 생성
def load_cooking_pdf_files(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    # 요리 문서에 최적화된 청킹
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 레시피는 보통 짧으므로 청크 크기 축소
        chunk_overlap=150
    )
    split_docs = text_splitter.split_documents(all_documents)

    vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    retriever = vector.as_retriever(search_kwargs={"k": 4})  # 관련 레시피 4개 검색

    retriever_tool = create_retriever_tool(
        retriever,
        name="recipe_book_search",
        description="업로드된 요리책이나 레시피 PDF에서 요리 정보를 검색할 때 사용합니다."
    )
    return retriever_tool

# ✅ 주방장 Agent 대화 실행
def chat_with_chef(user_input, agent_executor):
    result = agent_executor.invoke({"input": user_input})
    return result['output']

# ✅ 세션별 히스토리 관리
def get_session_history(session_ids):
    if session_ids not in st.session_state.session_history:
        st.session_state.session_history[session_ids] = ChatMessageHistory()
    return st.session_state.session_history[session_ids]

# ✅ 이전 메시지 출력
def print_messages():
    for msg in st.session_state["messages"]:
        with st.chat_message(msg['role']):
            if msg['role'] == 'assistant':
                st.markdown(f"👨‍🍳 **셰프 쿡쿡이**: {msg['content']}")
            else:
                st.markdown(f"🙋‍♂️ **고객님**: {msg['content']}")

# ✅ 메인 실행
def main():
    st.set_page_config(
        page_title="요리 비서 쿡쿡이", 
        layout="wide", 
        page_icon="👨‍🍳"
    )

    # 헤더 섹션
    col1, col2 = st.columns([1, 3])
    with col1:
        # 요리 비서 캐릭터 이미지 표시 (있다면)
        try:
            st.image('./chef_logo.png', width=200)
        except:
            st.markdown("👨‍🍳")
    
    with col2:
        st.title("🍽️ 요리 비서 '쿡쿡이' 👨‍🍳✨")
        st.markdown("**어떤 요리든 물어보세요! 레시피, 조리법, 재료, 팁까지 모두 알려드립니다** 🥘")

    st.markdown('---')

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}

    # 사이드바 설정
    with st.sidebar:
        st.header("🔧 설정")
        st.session_state["OPENAI_API"] = st.text_input(
            "OpenAI API 키", 
            placeholder="OpenAI API 키를 입력하세요", 
            type="password"
        )
        st.session_state["SERPAPI_API"] = st.text_input(
            "SerpAPI 키", 
            placeholder="SerpAPI 키를 입력하세요", 
            type="password"
        )
        
        st.markdown('---')
        st.header("📚 요리책 업로드")
        st.markdown("요리책이나 레시피 PDF를 업로드하면 더 정확한 요리 정보를 제공할 수 있어요!")
        
        cooking_pdfs = st.file_uploader(
            "요리책 PDF 파일 업로드", 
            accept_multiple_files=True, 
            key="cooking_pdf_uploader",
            type=['pdf']
        )
        
        st.markdown('---')
        st.header("🥘 추천 질문")
        st.markdown("""
        - "파스타 알리오 올리오 만드는 법"
        - "스테이크 굽는 온도와 시간"
        - "김치찌개 황금 레시피"
        - "마카롱 만들기 실패하지 않는 팁"
        - "초보자도 쉬운 요리 추천"
        """)

    # ✅ API 키 입력 확인
    if st.session_state["OPENAI_API"] and st.session_state["SERPAPI_API"]:
        os.environ['OPENAI_API_KEY'] = st.session_state["OPENAI_API"]
        os.environ['SERPAPI_API_KEY'] = st.session_state["SERPAPI_API"]

        # 도구 정의
        tools = []
        
        # PDF 요리책이 업로드된 경우 레시피 검색 도구 추가
        if cooking_pdfs:
            recipe_search = load_cooking_pdf_files(cooking_pdfs)
            tools.append(recipe_search)
            
        # 요리 전문 웹 검색 도구 추가
        tools.append(search_cooking_web())

        # LLM 설정 (요리 전문가 모드)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # 창의성을 위해 temperature 약간 높임

        # 20년 경력 호텔 주방장 프롬프트
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 20년 경력의 베테랑 호텔 주방장 '쿡쿡이'입니다. 
            
            **당신의 전문 분야:**
            - 한식, 양식, 중식, 일식, 프렌치 등 모든 요리 장르
            - 미슐랭 스타 레스토랑 수준의 고급 요리부터 가정 요리까지
            - 재료 선택, 조리법, 플레이팅, 와인 페어링
            - 요리 실패 해결법과 프로 팁
            
            **말투와 성격:**
            - 친근하고 따뜻한 멘토 같은 말투
            - 전문 지식을 쉽게 설명하는 능력
            - 요리에 대한 열정과 애정이 가득
            - 항상 이모지와 함께 생동감 있게 설명
            
            **응답 규칙:**
            1. 항상 한국어로 답변하세요
            2. 요리 관련 질문은 recipe_book_search 도구를 먼저 사용하세요
            3. PDF에서 정보를 찾을 수 없으면 cooking_search 도구를 사용하세요
            4. 레시피는 재료 → 조리과정 → 프로 팁 순서로 설명하세요
            5. 항상 이모지를 사용하여 친근하게 답변하세요
            6. 요리 초보자도 이해할 수 있도록 자세히 설명하세요
            
            **첫 인사말은 다음과 같이 시작하세요:**
            "안녕하세요! 요리 비서 '쿡쿡이'입니다! 👨‍🍳✨"
            """),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # 에이전트 생성
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True
        )

        # 채팅 인터페이스
        st.markdown("### 💬 쿡쿡이와 요리 상담")
        
        # 초기 인사말
        if not st.session_state["messages"]:
            initial_message = """
            안녕하세요! 요리 비서 '쿡쿡이'입니다! 👨‍🍳✨

            🍽️ **무엇을 도와드릴까요?**
            - 만들고 싶은 요리의 레시피가 궁금하시나요?
            - 요리 실패를 해결하고 싶으시나요?
            - 재료 활용법이나 조리 팁이 필요하시나요?

            어떤 요리든 편하게 물어보세요! 
            정성껏 알려드리겠습니다! 🥘💕
            """
            st.session_state["messages"].append({
                "role": "assistant", 
                "content": initial_message
            })

        # 이전 메시지들 출력
        print_messages()

        # 사용자 입력
        user_input = st.chat_input('요리에 대해 무엇이든 물어보세요! 🍳')

        if user_input:
            session_id = "cooking_session"
            session_history = get_session_history(session_id)

            # 사용자 메시지 추가
            st.session_state["messages"].append({
                "role": "user", 
                "content": user_input
            })

            # 주방장 응답 생성
            with st.spinner("쿡쿡이가 레시피를 찾고 있어요... 👨‍🍳"):
                try:
                    if session_history.messages:
                        # 이전 대화 맥락 포함
                        prev_msgs = [{
                            "role": "human" if msg.type == "human" else "assistant", 
                            "content": msg.content
                        } for msg in session_history.messages[-6:]]  # 최근 3턴만 유지
                        
                        context = f"이전 대화 맥락: {prev_msgs}"
                        response = chat_with_chef(f"{user_input}\n\n{context}", agent_executor)
                    else:
                        response = chat_with_chef(user_input, agent_executor)

                    # 응답 메시지 추가
                    st.session_state["messages"].append({
                        "role": "assistant", 
                        "content": response
                    })

                    # 세션 히스토리 업데이트
                    session_history.add_user_message(user_input)
                    session_history.add_ai_message(response)

                except Exception as e:
                    error_msg = f"죄송해요! 잠시 문제가 생겼네요. 🙏\n다시 한 번 질문해 주시겠어요? \n\n오류: {str(e)}"
                    st.session_state["messages"].append({
                        "role": "assistant", 
                        "content": error_msg
                    })

            # 페이지 새로고침하여 새 메시지 표시
            st.rerun()

    else:
        st.warning("⚠️ OpenAI API 키와 SerpAPI 키를 모두 입력해야 쿡쿡이와 대화할 수 있어요!")
        st.info("""
        **API 키 발급 방법:**
        1. **OpenAI API**: https://platform.openai.com/api-keys
        2. **SerpAPI**: https://serpapi.com/manage-api-key
        
        무료 크레딧으로도 충분히 사용 가능해요! 🎉
        """)

if __name__ == "__main__":
    main()
