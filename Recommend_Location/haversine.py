from math import radians, sin, cos, sqrt, atan2

class Haversine:
    def __init__(self, lat1, lon1, lat2, lon2):
        self.R = 6371000  # 지구 반지름 (단위: 미터)
        self.lat1, self.lon1, self.lat2, self.lon2 = map(radians, [lat1, lon1, lat2, lon2])
        self.dlat = self.lat2 - self.lat1
        self.dlon = self.lon2 - self.lon1

    def calculate(self):
        a = sin(self.dlat / 2)**2 + cos(self.lat1) * cos(self.lat2) * sin(self.dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return self.R * c  # 거리 (미터)