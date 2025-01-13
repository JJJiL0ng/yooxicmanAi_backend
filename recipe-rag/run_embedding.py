from rag.embedding import RecipeVectorStore
from langchain_community.document_loaders import Docx2txtLoader
from rag.embedding import RecipeVectorStore
def embed_recipes():
    try:
        print("\n=== 레시피 임베딩 프로세스 시작 ===")
        
        # 1. Vector Store 초기화
        print("\n1. Vector Store 초기화 중...")
        vector_store = RecipeVectorStore()
        
        # 2. Pinecone 인덱스 초기화
        print("\n2. Pinecone 인덱스 초기화 중...")
        if not vector_store.initialize_index():
            print("❌ 인덱스 초기화 실패")
            return
        print("✅ 인덱스 초기화 완료")
        
        # 3. DOCX 파일 로드
        print("\n3. 레시피 문서 로드 중...")
        file_path = "data/raw/test_raw_data.docx"
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        print(f"✅ 문서 로드 완료 (총 {len(documents)}개 문서)")
        
        # 4. 문서 임베딩
        print("\n4. 문서 임베딩 중...")
        success = vector_store.add_documents(documents)
        
        if success:
            print("✅ 임베딩 완료")
            
            # 5. 결과 확인
            print("\n5. 최종 상태 확인")
            stats = vector_store.index.describe_index_stats()
            print(f"- 총 벡터 수: {stats.total_vector_count}")
            
            # 6. 테스트 검색
            print("\n6. 테스트 검색 수행")
            results = vector_store.similarity_search("김치찌개", k=1)
            if results:
                print("\n검색 결과:")
                print(f"- 제목: {results[0].metadata.get('title', '제목 없음')}")
        else:
            print("❌ 임베딩 실패")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    embed_recipes()