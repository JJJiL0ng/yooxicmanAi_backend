from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import time
import re

class RecipeVectorStore:
    def __init__(self):
        load_dotenv()
        
        # Pinecone 초기화
        self.pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )
        self.index_name = "yooxicman-ai"
        
        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        
        # 텍스트 분할기 설정 수정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # 청크 크기 증가
            chunk_overlap=300,  # 오버랩 증가
            separators=["---", "\n레시피명:", "\nvideo_url:", "\n[", "\n#태그:","---"],  # 구분자 수정
            length_function=len,
            add_start_index=True,
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
        # 레시피를 개별 문서로 분리
        recipes = content.split("---")
        recipes = [recipe.strip() for recipe in recipes if recipe.strip()]
        
        metadata = {}
        
        for recipe in recipes:
            # 레시피명 추출
            title_match = re.search(r'레시피명:\s*(.+?)(?=\n|$)', recipe)
            metadata['title'] = title_match.group(1).strip() if title_match else "제목 없음"
            
            # 비디오 URL 추출
            url_match = re.search(r'video_url:\s*(https?://[^\s]+)', recipe)
            metadata['youtube_url'] = url_match.group(1).strip() if url_match else ""
            
            # 재료 섹션들 추출 (섹션별로 구분하여 저장)
            ingredients_list = []  # 변경: 딕셔너리 대신 리스트 사용
            sections = re.findall(r'\[(.*?)\](.*?)(?=\[|#태그:|$)', recipe, re.DOTALL)
            for section_name, section_content in sections:
                # 섹션 이름과 함께 재료를 리스트에 추가
                section_ingredients = [
                    f"{section_name.strip()}: {item.strip()}"
                    for item in section_content.strip().split(',')
                    if item.strip()
                ]
                ingredients_list.extend(section_ingredients)
            
            metadata['ingredients'] = ingredients_list  # 변경: 단순 리스트로 저장
            
            # 태그 추출
            tags_match = re.search(r'#태그:\s*(.+?)(?=$)', recipe)
            metadata['tags'] = [
                tag.strip() 
                for tag in tags_match.group(1).split(',')
            ] if tags_match else []
        
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
            
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 3
    ) -> List[tuple[Document, float]]:
        """유사도 점수와 함께 문서 검색"""
        try:
            vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings
            )
            
            # 기본 검색 결과 가져오기
            results = vector_store.similarity_search_with_score(query, k=k*2)
            
            # 결과 후처리 및 순위 조정
            processed_results = []
            query_terms = set(query.lower().split())
            
            for doc, score in results:
                # 메타데이터에서 태그와 재료 추출
                metadata = doc.metadata
                tags = set(tag.lower() for tag in metadata.get('tags', []))
                ingredients = set(
                    ing.lower().split(':')[-1].strip() 
                    for ing in metadata.get('ingredients', [])
                )
                
                # 태그 매칭 점수 계산
                tag_match = len(query_terms & tags) / len(query_terms) if query_terms else 0
                
                # 재료 매칭 점수 계산
                ingredient_match = len(query_terms & ingredients) / len(query_terms) if query_terms else 0
                
                # 최종 점수 계산 (벡터 유사도 + 태그 매칭 + 재료 매칭)
                final_score = (float(score) + tag_match + ingredient_match) / 3
                
                processed_results.append((doc, final_score))
            
            # 최종 점수로 정렬하고 상위 k개 반환
            processed_results.sort(key=lambda x: x[1], reverse=True)
            return processed_results[:k]
            
        except Exception as e:
            print(f"검색 중 에러 발생: {str(e)}")
            return []

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """유사도 기반 문서 검색"""
        results = self.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in results]