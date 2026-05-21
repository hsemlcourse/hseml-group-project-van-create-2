from typing import Literal, Optional

from pydantic import BaseModel, Field


class DeviceFeatures(BaseModel):
    device_type: Literal["smartphone", "laptop"]
    brand: str
    os: Optional[str] = None
    screen_inch: Optional[float] = Field(None, gt=0)
    storage_gb: Optional[float] = Field(None, gt=0)
    ram_gb: Optional[float] = Field(None, gt=0)
    battery_mah: Optional[float] = Field(None, gt=0)
    weight_g: Optional[float] = Field(None, gt=0)
    rear_camera_mp: Optional[float] = Field(None, ge=0)
    front_camera_mp: Optional[float] = Field(None, ge=0)
    release_year: Optional[int] = Field(None, ge=2000, le=2030)
    days_used: Optional[float] = Field(None, ge=0)
    condition: Optional[Literal["excellent", "good", "fair", "poor"]] = None
    weight_kg: Optional[float] = Field(None, gt=0)
    cpu_brand: Optional[str] = None
    gpu_brand: Optional[str] = None
    laptop_type: Optional[str] = None
    storage_type: Optional[str] = None


class PredictionResponse(BaseModel):
    price_rub: float
    price_rub_formatted: str
