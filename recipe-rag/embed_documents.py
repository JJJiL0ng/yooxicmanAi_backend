import asyncio
from dotenv import load_dotenv
from src.rag.embedding import RecipeVectorStore
from src.rag.chain import RecipeBot

async def embed_documents():
    # 환경 변수 로드
    load_dotenv()
    
    # 벡터 스토어 초기화
    vector_store = RecipeVectorStore()
    success = vector_store.initialize_index()
    
    if not success:
        print("벡터 스토어 초기화 실패")
        return
        
    # RecipeBot 초기화
    recipe_bot = RecipeBot(vector_store)
    
    # DOCX 파일 처리
    file_path = "data/raw/test_raw_data.docx"
    result = await recipe_bot.process_docx(file_path)
    
    if result:
        print("문서 임베딩 성공!")
    else:
        print("문서 임베딩 실패")

if __name__ == "__main__":
    asyncio.run(embed_documents())