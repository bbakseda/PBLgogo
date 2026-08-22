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
