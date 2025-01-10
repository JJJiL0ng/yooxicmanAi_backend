# src/config/config_loader.py
from dotenv import load_dotenv
import os
import yaml

class ConfigLoader:
    @staticmethod
    def load_config():
        # .env 파일 로드
        load_dotenv()
        
        # config.yaml 파일 로드
        with open("config/config.yaml") as f:
            config = yaml.safe_load(f)
            
        # 환경 변수 추가
        config['api']['openai']['api_key'] = os.getenv('OPENAI_API_KEY')
        config['vector_store']['pinecone']['api_key'] = os.getenv('PINECONE_API_KEY')
        config['vector_store']['pinecone']['environment'] = os.getenv('PINECONE_ENV')
        
        return config