# src/rag/chain.py
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from langchain_core.documents import Document

class RecipeBot:
    def __init__(self, vector_store: PineconeVectorStore):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7)
        self.vector_store = vector_store
        
        # 간단한 프롬프트 템플릿
        self.recipe_prompt = ChatPromptTemplate.from_messages([
            ("system", """주어진 재료로 만들 수 있거나 유사한 레시피의 재료 목록과 유튜브 링크를 찾아주세요.
답변 형식:
1. 레시피 이름
2. 필요한 재료 목록
3. 유튜브 링크

주의사항:
- 사용자가 가진 재료와 가장 많이 일치하는 레시피 우선
- 레시피 과정은 제외하고 재료 목록만 제공
- 모든 레시피는 반드시 유튜브 링크 포함"""),
            ("user", "{query}"),
            ("system", "참고할 레시피 정보:\n{context}"),
        ])
        
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """문서 내용에서 메타데이터 추출"""
        print("\n=== 메타데이터 추출 시작 ===")
        print(f"원본 문서 내용:\n{content}\n")
        
        # 제목 추출
        title_match = re.search(r'레시피명:\s*(.+?)(?=\n|$)', content)
        title = title_match.group(1).strip() if title_match else "제목 없음"
        print(f"추출된 제목: {title}")
        
        # URL 추출
        url_match = re.search(r'veido_url\s*\[URL\]:\s*(.+?)(?=\n|$)', content)
        youtube_url = url_match.group(1).strip() if url_match else ""
        print(f"추출된 URL: {youtube_url}")
        
        # 재료 추출
        ingredients = []
        ingredients_section = re.search(r'\[완성재료/매인재료\](.*?)(?=\[|$)', content, re.DOTALL)
        if ingredients_section:
            ingredients_text = ingredients_section.group(1)
            # 줄바꿈으로 구분된 재료들을 리스트로 변환
            ingredients = [
                ing.strip() 
                for ing in ingredients_text.split('\n') 
                if ing.strip() and not ing.startswith('[')
            ]
        print(f"추출된 재료: {ingredients}")
        
        # 다른 형식의 재료 섹션 시도
        if not ingredients:
            ingredients_alt = re.findall(r'(?:^|\n)([^[\n].*?)(?=\n|$)', content)
            ingredients = [ing.strip() for ing in ingredients_alt if ing.strip()]
            print(f"대체 방식으로 추출된 재료: {ingredients}")
        
        metadata = {
            "title": title,
            "youtube_url": youtube_url,
            "ingredients": ingredients
        }
        print(f"최종 메타데이터: {metadata}\n")
        
        return metadata
            
    async def process_docx(self, file_path: str) -> bool:
        """DOCX 파일에서 재료 목록과 유튜브 링크 추출"""
        try:
            loader = Docx2txtLoader(file_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            )
            documents = loader.load_and_split(text_splitter=text_splitter)
            
            # 벡터 스토어에 문서 추가
            self.vector_store.add_documents(documents)
            return True
        except Exception as e:
            print(f"Error processing DOCX: {str(e)}")
            return False
            
    async def find_matching_recipes(
        self,
        ingredients: List[str],
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """주어진 재료와 매칭되는 레시피 검색"""
        try:
            query = f"재료: {', '.join(ingredients)}"
            print(f"\n=== 레시피 검색 시작 ===")
            print(f"검색 쿼리: {query}")
            
            results = self.vector_store.similarity_search_with_score(query, k=k)
            print(f"검색된 문서 수: {len(results)}")
            
            matching_recipes = []
            for i, (doc, score) in enumerate(results):
                print(f"\n문서 {i+1}:")
                print(f"유사도 점수: {score}")
                print(f"문서 내용:\n{doc.page_content}")
                
                metadata = self.extract_metadata(doc.page_content)
                recipe_info = {
                    "title": metadata["title"],
                    "youtube_url": metadata["youtube_url"],
                    "ingredients": metadata["ingredients"],
                    "similarity_score": float(score)
                }
                matching_recipes.append(recipe_info)
                print(f"추출된 레시피 정보: {recipe_info}")
                    
            print(f"\n최종 매칭된 레시피 수: {len(matching_recipes)}")
            return matching_recipes
            
        except Exception as e:
            print(f"레시피 검색 중 에러 발생: {str(e)}")
            raise Exception(f"Error finding matching recipes: {str(e)}")

# src/api/main.py
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI(title="Recipe Matching Bot")

class IngredientRequest(BaseModel):
    ingredients: List[str]
    limit: Optional[int] = 3

class RecipeMatch(BaseModel):
    title: str
    youtube_url: str
    ingredients: List[str]
    similarity_score: float

class RecipeMatchResponse(BaseModel):
    matches: List[RecipeMatch]

@app.post("/recipes/match", response_model=RecipeMatchResponse)
async def match_recipes(request: IngredientRequest):
    """주어진 재료와 매칭되는 레시피 검색"""
    try:
        matches = await recipe_bot.find_matching_recipes(
            ingredients=request.ingredients,
            k=request.limit
        )
        return {"matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recipes/load")
async def load_recipe_document(file_path: str):
    """레시피 문서 로드"""
    try:
        success = await recipe_bot.process_docx(file_path)
        if success:
            return {"message": "Recipe document loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load document")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 테스트 스크립트
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    import os
    
    async def test_recipe_matching():
        load_dotenv()
        
        # 테스트용 재료 목록
        test_ingredients = ["소고기", "양파", "당근"]
        
        request = IngredientRequest(ingredients=test_ingredients)
        response = await match_recipes(request)
        
        print("\n=== 테스트 결과 ===")
        print(f"검색한 재료: {', '.join(test_ingredients)}")
        print("\n매칭된 레시피:")
        for match in response.matches:
            print(f"\n제목: {match.title}")
            print(f"유튜브 링크: {match.youtube_url}")
            print(f"필요한 재료: {', '.join(match.ingredients)}")
            print(f"유사도 점수: {match.similarity_score:.2f}")
            
    asyncio.run(test_recipe_matching())