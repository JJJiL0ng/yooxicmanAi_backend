# src/api/main.py
from fastapi import FastAPI, HTTPException, Request
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
async def match_recipes(request: IngredientRequest, req: Request):
    """주어진 재료와 매칭되는 레시피 검색"""
    try:
        recipe_bot = req.app.state.recipe_bot
        matches = await recipe_bot.find_matching_recipes(
            ingredients=request.ingredients,
            k=request.limit
        )
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