# run.py
import uvicorn
from dotenv import load_dotenv
from src.rag.embedding import RecipeVectorStore
from src.rag.chain import RecipeBot
from src.api.main import app
import os

def init_rag_components():
    """RAG 컴포넌트 초기화"""
    # 환경 변수 로드
    load_dotenv()
    
    # Pinecone 벡터 스토어 초기화
    vector_store = RecipeVectorStore()
    success = vector_store.initialize_index()
    
    if not success:
        print("Vector store initialization failed. Check your Pinecone settings.")
        raise Exception("Failed to initialize vector store")
        
    # Recipe Bot 초기화
    recipe_bot = RecipeBot(vector_store)
    return recipe_bot

# 전역 변수로 recipe_bot 설정
recipe_bot = init_rag_components()
app.state.recipe_bot = recipe_bot

def main():
    try:
        # 서버 설정
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        
        # 서버 실행
        print(f"Starting server on {host}:{port}")
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=False  # reload를 False로 변경
        )
        
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        raise

if __name__ == "__main__":
    main()