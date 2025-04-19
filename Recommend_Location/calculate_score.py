class CalculateScore:
    def __init__(self, rating, distance, vibe_similarity, max_distance):
        self.rating = rating
        self.distance = distance
        self.vibe_similarity = vibe_similarity
        self.max_distance = max_distance
    
    def calculate(self):
        rating_score = max(0, (self.rating - 2.5))  # 2.5 기준
        distance_score = max(0, (self.max_distance - self.distance) / self.max_distance)
        final_score = rating_score * 0.2 + distance_score * 0.3 + self.vibe_similarity * 0.5
        return final_score