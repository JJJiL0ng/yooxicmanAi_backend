# src/preprocessing/validator.py
from typing import List

class RecipeValidator:
    def check_recipe_format(self, recipe: dict) -> bool:
        required_fields = ['title', 'ingredients', 'steps']
        return all(field in recipe for field in required_fields)
        
    def check_ingredients(self, ingredients: List[str]) -> bool:
        return all(len(ing.strip()) > 0 for ing in ingredients)
        
    def check_steps(self, steps: List[str]) -> bool:
        return all(len(step.strip()) > 0 for step in steps)