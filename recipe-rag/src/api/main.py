# src/api/main.py
from fastapi import FastAPI, HTTPException, Request
from typing import List, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Recipe Matching Bot")

# CORS 설정 수정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

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

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/recipes/match", response_model=RecipeMatchResponse)
async def match_recipes(request: IngredientRequest, req: Request):
    """주어진 재료와 매칭되는 레시피 검색"""
    try:
        recipe_bot = req.app.state.recipe_bot
        matches = await recipe_bot.find_matching_recipes(
            ingredients=request.ingredients,
            k=request.limit
        )
        print("\n=== 프론트엔드로 전송되는 레시피 매칭 결과 ===")
        print(f"검색된 레시피 수: {len(matches)}")
        for i, match in enumerate(matches, 1):
            print(f"\n[레시피 {i}]")
            print(f"제목: {match['title']}")
            print(f"유사도 점수: {match['similarity_score']:.4f}")
            print(f"YouTube URL: {match['youtube_url']}")
            print("재료:")
            for section, ingredients in match['ingredients'].items():
                print(f"  [{section}]")
                for ing in ingredients:
                    print(f"    - {ing}")
        print("\n" + "="*50)
        return {"matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recipes/load")
async def load_recipe_document(file_path: str, req: Request):
    """레시피 문서 로드"""
    try:
        recipe_bot = req.app.state.recipe_bot
        success = await recipe_bot.process_docx(file_path)
        if success:
            return {"message": "Recipe document loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load document")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """사용자 메시지에 대한 챗봇 응답"""
    try:
        recipe_bot = req.app.state.recipe_bot
        if recipe_bot is None:
            raise HTTPException(status_code=500, detail="Recipe bot is not initialized")
            
        if not request.message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
            
        response = await recipe_bot.chat(request.message)
        print("\n=== 프론트엔드로 전송되는 채팅 응답 ===")
        print(f"사용자 메시지: {request.message}")
        print("\n응답 내용:")
        print(response)
        print("\n" + "="*50)
        
        if not response:
            raise HTTPException(status_code=500, detail="Failed to generate response")
            
        return {"response": response}
    except Exception as e:
        print(f"Chat error: {str(e)}")  # 서버 로그에 에러 출력
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")