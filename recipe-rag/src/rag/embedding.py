# src/rag/embedding.py
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import time

class RecipeVectorStore:
    def __init__(self):
        load_dotenv()
        
        # Pinecone 초기화
        self.pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )
        self.index_name = "yooxicmanrag"
        
        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        
        # 텍스트 분할기 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        
    def initialize_index(self) -> bool:
        """Pinecone 인덱스 초기화"""
        try:
            # 기존 인덱스 확인
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"새로운 인덱스 '{self.index_name}' 생성 중...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric='cosine'
                )
                # 인덱스가 준비될 때까지 대기
                while not self.pc.describe_index(self.index_name).status['ready']:
                    print("인덱스 준비 중...")
                    time.sleep(1)
            
            self.index = self.pc.Index(self.index_name)
            return True
            
        except Exception as e:
            print(f"Error initializing index: {str(e)}")
            return False
            
    def process_document(self, content: str) -> Dict[str, Any]:
        """문서 내용에서 메타데이터 추출"""
        print("\n=== 문서 처리 시작 ===")
        print(f"원본 내용:\n{content}\n")
        
        # 제목 추출
        title = "제목 없음"
        if "레시피명:" in content:
            title_lines = [line for line in content.split('\n') if "레시피명:" in line]
            if title_lines:
                title = title_lines[0].replace("레시피명:", "").strip()
        
        # URL 추출
        youtube_url = ""
        if "veido_url [URL]:" in content:
            url_lines = [line for line in content.split('\n') if "veido_url [URL]:" in line]
            if url_lines:
                youtube_url = url_lines[0].replace("veido_url [URL]:", "").strip()
        
        # 재료 추출
        ingredients = []
        if "[완성재료/매인재료]" in content:
            parts = content.split("[완성재료/매인재료]")
            if len(parts) > 1:
                ingredients_text = parts[1].split("[")[0]  # 다음 섹션 전까지
                ingredients = [
                    ing.strip() 
                    for ing in ingredients_text.split('\n') 
                    if ing.strip()
                ]
        
        metadata = {
            "title": title,
            "youtube_url": youtube_url,
            "ingredients": ingredients
        }
        print(f"추출된 메타데이터: {metadata}")
        return metadata

    def add_documents(self, documents: List[Document]) -> bool:
        """문서를 벡터 스토어에 추가"""
        try:
            processed_docs = []
            for doc in documents:
                # 메타데이터 추출
                metadata = self.process_document(doc.page_content)
                # 새 Document 객체 생성
                processed_doc = Document(
                    page_content=doc.page_content,
                    metadata=metadata
                )
                processed_docs.append(processed_doc)
            
            # 청크 생성
            chunks = self.text_splitter.split_documents(processed_docs)
            print(f"\n생성된 청크 수: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"\n청크 {i+1} 메타데이터: {chunk.metadata}")
            
            # 벡터 스토어에 저장
            vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings
            )
            vector_store.add_documents(chunks)
            
            # 저장 확인
            stats = self.index.describe_index_stats()
            print(f"\n현재 인덱스 상태:")
            print(f"- 총 벡터 수: {stats.total_vector_count}")
            
            return True
            
        except Exception as e:
            print(f"문서 추가 중 에러 발생: {str(e)}")
            return False
            
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple[Document, float]]:
        """유사도 점수와 함께 문서 검색"""
        vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings
        )
        return vector_store.similarity_search_with_score(query, k=k)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """유사도 기반 문서 검색"""
        vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings
        )
        return vector_store.similarity_search(query, k=k)