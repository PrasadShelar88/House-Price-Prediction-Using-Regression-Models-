from pydantic import BaseModel, Field

class HouseInput(BaseModel):
    area_sqft: float = Field(1800, ge=250, le=10000)
    bedrooms: int = Field(3, ge=1, le=10)
    bathrooms: int = Field(2, ge=1, le=8)
    location: str = Field("Urban")
    property_age: int = Field(8, ge=0, le=100)
    parking: int = Field(1, ge=0, le=5)
    furnishing: str = Field("Semi-Furnished")
    floors: int = Field(2, ge=1, le=6)
    has_garden: str = Field("No")
    distance_to_city_km: float = Field(7.5, ge=0, le=80)

class PredictionResponse(BaseModel):
    predicted_price_inr: float
    predicted_price_lakh: float
    model_name: str
    confidence_note: str
