# src/preprocessing/formatter.py
import re
from typing import Dict, List

class RecipeFormatter:
    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
        
    def extract_metadata(self, text: str) -> Dict:
        # 레시피 제목, 조리시간, 난이도 등 메타데이터 추출
        pass
        
    def structure_recipe(self, text: str) -> Dict:
        # 재료와 조리 단계를 구조화된 형태로 변환
        pass