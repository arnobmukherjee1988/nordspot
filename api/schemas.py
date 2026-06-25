"""
Pydantic request/response schemas for the NordSpot API.

Implemented in Epic 5.
"""
# from pydantic import BaseModel
# from datetime import date
# from typing import List

# class ForecastRequest(BaseModel):
#     zone: str           # e.g. "SE3"
#     forecast_date: date # e.g. "2026-06-25"

# class HourlyForecast(BaseModel):
#     hour: int           # 0–23
#     price_eur_mwh: float
#     lower_bound: float
#     upper_bound: float

# class ForecastResponse(BaseModel):
#     zone: str
#     forecast_date: date
#     hours: List[HourlyForecast]
