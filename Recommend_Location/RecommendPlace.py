from dotenv import load_dotenv
import os
import requests
import torch
import torch.nn.functional as F
from extract_mbti_keywords import ExtractMBTIKeywords
from get_reviews_embedding import GetReviewEmbedding
from compute_similarity_score import ComputeSimilarityScore
from calculate_score import CalculateScore
from haversine import Haversine # 직접 정의했다면 클래스 가져오기

class RecommendPlace:
    def __init__(self, model, embedding_model, mbti_vector):
        load_dotenv()
        self.API_KEY = os.getenv("API_KEY")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.embedding_model = embedding_model

        self.keywords = ExtractMBTIKeywords().extract(mbti_vector)

    def recommend(self, query, lat, lng, radius, cutoff_days):
        recommendations = []

        query = query.replace(" ", "+")
        url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={query}&location={lat},{lng}&radius={radius}&language=ko&key={self.API_KEY}"

        response = requests.get(url)
        data = response.json()

        keyword_vectors = [self.embedding_model.encode(k, convert_to_tensor=True) for k in self.keywords]
        mbti_input_tensor = F.normalize(torch.stack(keyword_vectors).mean(dim=0, keepdim=True), dim=1).to(self.device)
        user_vibe = F.normalize(self.model(mbti_input_tensor), dim=1).detach().cpu().numpy()

        for place in data.get("results", []):
            name = place.get("name")
            address = place.get("formatted_address")
            rating = place.get("rating", 0)
            place_id = place.get("place_id")
            location = place.get("geometry", {}).get("location", {})
            lat_p, lng_p = location.get("lat"), location.get("lng")

            distance = Haversine(lat, lng, lat_p, lng_p).calculate()
            review_texts = []

            # 상세 리뷰 가져오기
            detail_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&language=ko&key={self.API_KEY}"
            detail_data = requests.get(detail_url).json()
            reviews = detail_data.get("result", {}).get("reviews", [])

            for review in reviews:
                time_desc = review.get("relative_time_description", "")
                days = 9999
                try:
                    if "일 전" in time_desc:
                        days = int(time_desc.replace("일 전", "").strip())
                    elif "주일 전" in time_desc or "주 전" in time_desc:
                        week_str = time_desc.replace("주일 전", "").replace("주 전", "").strip()
                        days = int(week_str) * 7 if week_str.isdigit() else 9999
                    elif "개월 전" in time_desc or "달 전" in time_desc:
                        month_str = time_desc.replace("개월 전", "").replace("달 전", "").strip()
                        days = int(month_str) * 30 if month_str.isdigit() else 9999
                    elif "년 전" in time_desc:
                        year_str = time_desc.replace("년 전", "").strip()
                        days = int(year_str) * 365 if year_str.isdigit() else 9999
                    elif "이내" in time_desc:
                        days = 0
                except Exception as e:
                    print(f"날짜 파싱 에러: {time_desc} → {e}")
                    days = 9999

                if days <= cutoff_days:
                    review_texts.append(review.get("text", ""))

            review_vector = GetReviewEmbedding(self.embedding_model).embedding(review_texts)
            vibe_similarity = ComputeSimilarityScore(self.model).compute(user_vibe, review_vector)
            score = CalculateScore(rating, distance, vibe_similarity, radius).calculate()

            recommendations.append({
                "name": name,
                "address": address,
                "rating": rating,
                "distance": distance,
                "similarity": vibe_similarity,
                "score": score
            })

        recommendations_sorted = sorted(recommendations, key=lambda x: x["score"], reverse=True)
        return recommendations_sorted