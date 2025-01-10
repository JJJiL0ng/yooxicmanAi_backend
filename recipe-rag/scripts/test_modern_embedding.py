# scripts/test_modern_embedding.py
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import time

def test_document_embedding():
    # 환경 변수 로드
    load_dotenv()
    
    try:
        # 1. 문서 로드 및 분할
        print("1. 문서 로드 중...")
        loader = Docx2txtLoader('data/raw/raw_test.docx')
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
        )
        
        document_list = loader.load_and_split(text_splitter=text_splitter)
        print(f"- 생성된 문서 청크 수: {len(document_list)}")
        print(f"- 첫 번째 청크 미리보기:\n{document_list[0].page_content[:200]}...")
        
        # 2. 임베딩 설정
        print("\n2. 임베딩 모델 초기화...")
        embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        
        # 3. Pinecone 초기화
        print("\n3. Pinecone 초기화...")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=pinecone_api_key)
        
        # 인덱스 이름 설정
        index_name = "yooxicmanrag"
        
        # 기존 인덱스 확인 또는 생성
        existing_indexes = [index.name for index in pc.list_indexes()]
        if index_name not in existing_indexes:
            print(f"- 새로운 인덱스 '{index_name}' 생성 중...")
            pc.create_index(
                name=index_name,
                dimension=1536,  # text-embedding-3-large의 차원
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='gcp-starter'  # 무료 플랜용 리전으로 변경
                )
            )
            print("- 인덱스 생성 완료")
            
            # 인덱스가 준비될 때까지 대기
            while not pc.describe_index(index_name).status['ready']:
                print("- 인덱스 준비 중...")
                time.sleep(1)
        
        print("- 인덱스 준비 완료")
        index = pc.Index(index_name)
        
        # 4. 벡터 스토어에 문서 추가
        print("\n4. 벡터 스토어에 문서 저장 중...")
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        vector_store.add_documents(document_list)
        
        # 5. 검색 테스트
        print("\n5. 검색 테스트 수행...")
        query = "브리스킷(차돌양지)를 가지고 있는데 이걸 사용할 요리에 필요한 재료를 알려주세요"
        results = vector_store.similarity_search(query, k=1)
        
        print("\n검색 결과:")
        if results and len(results) > 0:
            print(f"- 가장 관련성 높은 문서 내용:\n{results[0].page_content[:300]}...")
        else:
            print("- 검색 결과가 없습니다. 다음을 확인해주세요:")
            print("  1. 인덱스에 데이터가 제대로 저장되었는지")
            print("  2. 검색 쿼리가 적절한지")
            print("  3. 임베딩이 정상적으로 생성되었는지")
            
        # 저장된 문서 수 확인
        stats = index.describe_index_stats()
        print(f"\n현재 인덱스 상태:")
        print(f"- 총 벡터 수: {stats.total_vector_count}")
        print(f"- 차원 수: {stats.dimension}")
        
        return True
        
    except Exception as e:
        print(f"\nError: 처리 중 오류 발생 - {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("최신 LangChain을 이용한 문서 임베딩 테스트 시작...\n")
    success = test_document_embedding()
    print(f"\n테스트 결과: {'성공' if success else '실패'}")