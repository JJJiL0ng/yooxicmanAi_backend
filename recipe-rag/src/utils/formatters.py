# src/utils/formatters.py
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class RecipeOutput:
    title: str
    youtube_url: str
    ingredients: List[str]
    tags: List[str]
    similarity_score: float

class RecipeFormatter:
    @staticmethod
    def parse_recipe_content(content: str) -> Dict:
        """레시피 문서 내용을 구조화된 형태로 파싱"""
        lines = content.split('\n')
        recipe = {
            "title": "",
            "youtube_url": "",
            "ingredients": [],
            "tags": []
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("레시피명:"):
                recipe["title"] = line.replace("레시피명:", "").strip()
            elif line.startswith(("video_url:", "veido_url:")):
                recipe["youtube_url"] = line.split(":", 1)[1].strip().replace("[URL]: ", "")
            elif line.startswith("#태그:"):
                recipe["tags"] = [tag.strip() for tag in line.replace("#태그:", "").split(",")]
            elif "," in line and not line.startswith(("#", "레시피명", "video_url", "veido_url")):
                ingredients = [ing.strip() for ing in line.split(",") if ing.strip()]
                recipe["ingredients"].extend(ingredients)
                
        return recipe

class OutputFormatter:
    @staticmethod
    def format_recipe_match(recipe_data: Dict) -> Optional[RecipeOutput]:
        """레시피 데이터를 출력 형식으로 변환"""
        if not recipe_data.get("title") or not recipe_data.get("youtube_url"):
            return None
            
        return RecipeOutput(
            title=recipe_data["title"],
            youtube_url=recipe_data["youtube_url"],
            ingredients=recipe_data.get("ingredients", []),
            tags=recipe_data.get("tags", []),
            similarity_score=recipe_data.get("similarity_score", 0.0)
        )
        
    @staticmethod
    def format_matches_response(matches: List[Dict]) -> Dict:
        """매칭된 레시피 목록을 응답 형식으로 변환"""
        formatted_matches = []
        seen_titles = set()
        
        for match in matches:
            recipe_output = OutputFormatter.format_recipe_match(match)
            if recipe_output and recipe_output.title not in seen_titles:
                seen_titles.add(recipe_output.title)
                formatted_matches.append({
                    "title": recipe_output.title,
                    "youtube_url": recipe_output.youtube_url,
                    "ingredients": recipe_output.ingredients,
                    "tags": recipe_output.tags,
                    "similarity_score": recipe_output.similarity_score
                })
                
        return {"matches": formatted_matches}