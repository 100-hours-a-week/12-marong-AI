from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from RecommendPlace import RecommendPlace
from extract_mbti_keywords import ExtractMBTIKeywords
from get_reviews_embedding import GetReviewEmbedding
from compute_similarity_score import ComputeSimilarityScore
from haversine import Haversine
from mbti_projector import MBTIProjector
import torch
import os
from sentence_transformers import SentenceTransformer
import json

app = FastAPI()

# ✅ 모델 로딩
mbti_model = MBTIProjector()
mbti_model.load_state_dict(torch.load("best_mbti_projector.pt", map_location="cpu"))
mbti_model.eval()

embedding_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# API 입력 형태 정의
class RecommendInput(BaseModel):
    user_id: str
    mbti_vector: List[float]  # 예: [0.5, 0.5, 0.5, 0.5]
    category: str  # 예: '한식'
    latitude: float
    longitude: float
    max_distance: int = 1000  # 미터 단위

@app.post("/recommend/place")
def recommend_places(input: RecommendInput):
    # 추천 시스템 초기화
    recommender = RecommendPlace(
        model=mbti_model,
        embedding_model=embedding_model,
        mbti_vector=input.mbti_vector
    )

    # 추천 수행
    results = recommender.recommend(
        input.category,
        input.latitude,
        input.longitude,
        input.max_distance,
        700
    )

    # 결과 구성
    return {
        "user_id": input.user_id,
        "message": "recommend_success",
        "data": [
            {
                "name": r["name"],
                "address": r["address"],
                "rating": r["rating"],
                "distance": round(r["distance"], 1)
            }
            for r in results[:5]  # 상위 5개
        ]
    }