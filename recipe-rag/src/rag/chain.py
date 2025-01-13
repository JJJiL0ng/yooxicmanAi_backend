from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
from collections import defaultdict

class RecipeBot:
    def __init__(self, vector_store: PineconeVectorStore):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.vector_store = vector_store
        
        # 프롬프트 템플릿 수정
        self.recipe_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 요리 레시피 전문가입니다. 사용자가 제공한 재료들을 바탕으로 가장 적합한 레시피를 추천해주세요.

주어진 문서에서 재료, 레시피명, 영상 링크가 서로 연관된 하나의 레시피만을 선택하여 답변해주세요.

답변 형식:
1. 레시피명: {title}
2. 필요한 재료:
   {ingredients}
3. 조리 영상: {youtube_url}

주의사항:
- 서로 다른 레시피의 정보를 혼합하지 마세요
- 문서에 있는 정보만을 사용하세요
- 재료 목록은 원본 문서의 섹션을 유지하여 표시하세요"""),
            ("user", "{query}"),
            ("system", "참고할 레시피 정보:\n{context}"),
        ])
        
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """문서 내용에서 메타데이터 추출"""
        metadata = {}
        
        # 레시피명 추출
        title_match = re.search(r'레시피명:\s*(.+?)(?=\n|$)', content)
        metadata['title'] = title_match.group(1).strip() if title_match else "제목 없음"
        
        # URL 추출
        url_match = re.search(r'video_url:\s*(https?://[^\s]+)', content)
        metadata['youtube_url'] = url_match.group(1).strip() if url_match else ""
        
        # 재료 섹션 추출
        ingredients_dict = defaultdict(list)
        sections = re.findall(r'\[(.*?)\](.*?)(?=\[|#태그:|$)', content, re.DOTALL)
        for section_name, section_content in sections:
            ingredients = [
                ing.strip()
                for ing in section_content.split(',')
                if ing.strip()
            ]
            ingredients_dict[section_name.strip()] = ingredients
        
        metadata['ingredients'] = dict(ingredients_dict)
        
        # 태그 추출 (검색용)
        tags_match = re.search(r'#태그:\s*(.+?)(?=---|$)', content)
        metadata['tags'] = [
            tag.strip() 
            for tag in tags_match.group(1).split(',')
        ] if tags_match else []
        
        return metadata
            
    async def process_docx(self, file_path: str) -> bool:
        """DOCX 파일 처리"""
        try:
            loader = Docx2txtLoader(file_path)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["---\n", "\n[", "\n#태그:", "\n", ","]
            )
            documents = loader.load_and_split(text_splitter=text_splitter)
            self.vector_store.add_documents(documents)
            return True
        except Exception as e:
            print(f"Error processing DOCX: {str(e)}")
            return False
    
    def calculate_search_score(
        self,
        query_terms: List[str],
        recipe_metadata: Dict[str, Any]
    ) -> float:
        """검색 점수 계산"""
        # 쿼리 용어를 소문자로 변환
        query_terms = [term.lower() for term in query_terms]
        
        # 재료 매칭 점수 계산
        all_ingredients = [ing.lower() for ing in recipe_metadata['ingredients']]
        
        ingredient_matches = sum(
            any(term in ing for ing in all_ingredients)
            for term in query_terms
        )
        ingredient_score = ingredient_matches / len(query_terms) if query_terms else 0
        
        # 태그 매칭 점수 계산
        tags = [tag.lower() for tag in recipe_metadata['tags']]
        tag_matches = sum(
            any(term in tag for tag in tags)
            for term in query_terms
        )
        tag_score = tag_matches / len(query_terms) if query_terms else 0
        
        # 레시피 이름 매칭 점수 계산
        title = recipe_metadata['title'].lower()
        title_matches = sum(term in title for term in query_terms)
        title_score = title_matches / len(query_terms) if query_terms else 0
        
        # 최종 점수 계산 (가중치 적용)
        final_score = (
            0.5 * ingredient_score +  # 재료 매칭이 가장 중요
            0.3 * tag_score +        # 태그 매칭
            0.2 * title_score        # 제목 매칭
        )
        
        return final_score

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
            
            # 벡터 검색으로 후보 가져오기
            results = self.vector_store.similarity_search_with_score(query, k=k*3)
            
            matching_recipes = []
            for doc, vector_score in results:
                metadata = self.extract_metadata(doc.page_content)
                
                # 검색 점수 계산
                search_score = self.calculate_search_score(ingredients, metadata)
                
                # 최종 점수 계산 (검색 점수 + 벡터 유사도)
                final_score = (0.7 * search_score) + (0.3 * float(vector_score))
                
                recipe_info = {
                    "title": metadata["title"],
                    "youtube_url": metadata["youtube_url"],
                    "ingredients": metadata["ingredients"],
                    "similarity_score": final_score
                }
                matching_recipes.append(recipe_info)
            
            # 최종 점수로 정렬
            matching_recipes.sort(key=lambda x: x["similarity_score"], reverse=True)
            return matching_recipes[:k]
            
        except Exception as e:
            print(f"레시피 검색 중 에러 발생: {str(e)}")
            raise Exception(f"Error finding matching recipes: {str(e)}")

    async def chat(self, message: str) -> str:
        """사용자 메시지에 대한 응답 생성"""
        try:
            # 재료 목록 추출
            ingredients = [
                ing.strip()
                for ing in message.replace("재료:", "").split(',')
                if ing.strip()
            ]
            
            # 레시피 검색
            recipes = await self.find_matching_recipes(ingredients)
            
            # 간단한 출력 형식으로 변환
            responses = []
            for i, recipe in enumerate(recipes, 1):
                # 모든 섹션의 재료를 하나의 리스트로 병합
                all_ingredients = []
                for section, items in recipe['ingredients'].items():
                    all_ingredients.extend(items)
                
                recipe_text = f"""레시피 {i}번째
레시피명: {recipe['title']}

[필요한 재료]
{chr(10).join(f'- {item}' for item in all_ingredients)}

유튜브 링크: {recipe['youtube_url']}
---"""
                responses.append(recipe_text)
            
            final_response = "\n\n".join(responses)
            return final_response
            
        except Exception as e:
            print(f"Error in chat method: {str(e)}")
            return "레시피 검색 중 오류가 발생했습니다."