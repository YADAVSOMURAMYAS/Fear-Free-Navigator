"""
api/models/request.py
=====================
Pydantic request models for all API endpoints.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional


class RouteRequest(BaseModel):
    origin_lat:  float = Field(..., ge=-90,  le=90,  description="Origin latitude")
    origin_lon:  float = Field(..., ge=-180, le=180, description="Origin longitude")
    dest_lat:    float = Field(..., ge=-90,  le=90,  description="Destination latitude")
    dest_lon:    float = Field(..., ge=-180, le=180, description="Destination longitude")
    alpha:       float = Field(0.7, ge=0.0,  le=1.0, description="Safety weight 0-1")
    hour:        int   = Field(22,  ge=0,    le=23,  description="Hour of day 0-23")

    @validator("alpha")
    def round_alpha(cls, v):
        return round(v, 2)

    class Config:
        json_schema_extra = {
            "example": {
                "origin_lat":  12.9767,
                "origin_lon":  77.6009,
                "dest_lat":    12.9352,
                "dest_lon":    77.6245,
                "alpha":       0.7,
                "hour":        22,
            }
        }


class ScoreRequest(BaseModel):
    lat:  float = Field(..., ge=-90,  le=90)
    lon:  float = Field(..., ge=-180, le=180)
    hour: int   = Field(22,  ge=0,    le=23)


class ReportRequest(BaseModel):
    lat:         float  = Field(..., ge=-90,  le=90)
    lon:         float  = Field(..., ge=-180, le=180)
    description: str    = Field(..., min_length=5, max_length=500)
    category:    str    = Field("unsafe", description="unsafe/harassment/poor_lighting/other")
    hour:        Optional[int] = Field(None, ge=0, le=23)

    @validator("category")
    def validate_category(cls, v):
        allowed = {"unsafe", "harassment", "poor_lighting", "crime", "other"}
        if v not in allowed:
            return "other"
        return v