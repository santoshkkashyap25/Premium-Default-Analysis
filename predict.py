"""FastAPI Production REST Service for Insurance Premium Default Risk Profiling.

Provides high-performance asynchronous endpoints for single and batch customer
default scoring, risk tier assignment, and intervention routing.
"""

from typing import List, Optional, Union
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from src.inference import InferencePipeline

# Initialize FastAPI App
app = FastAPI(
    title="Insurance Premium Default Risk Profiler API",
    description="Production REST service for predicting premium payment lapse and routing into tiered operational interventions.",
    version="2.0.0",
)

# Global pipeline instance initialized on import
try:
    pipeline: Optional[InferencePipeline] = InferencePipeline()
except Exception as e:
    pipeline: Optional[InferencePipeline] = None
    print(f"Warning: InferencePipeline initialization deferred: {e}")


class PolicyholderInput(BaseModel):
    id: Optional[str] = Field(default=None, description="Optional customer or policy identifier")
    perc_premium_paid_by_cash_credit: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of premium paid by cash or credit (0.0 to 1.0)"
    )
    age: Optional[int] = Field(default=None, ge=18, le=120, description="Age in years")
    age_in_days: Optional[int] = Field(default=None, ge=6500, description="Age in days (if age in years is not provided)")
    Income: float = Field(..., ge=0.0, description="Annual income of policyholder")
    Count_3_6_months_late: float = Field(
        default=0.0, ge=0.0, alias="Count_3-6_months_late", description="Number of times premium was 3-6 months late"
    )
    Count_6_12_months_late: float = Field(
        default=0.0, ge=0.0, alias="Count_6-12_months_late", description="Number of times premium was 6-12 months late"
    )
    Count_more_than_12_months_late: float = Field(
        default=0.0, ge=0.0, alias="Count_more_than_12_months_late", description="Number of times premium was >12 months late"
    )
    application_underwriting_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Underwriting score (0.0 to 100.0). Missing values will be imputed."
    )
    no_of_premiums_paid: int = Field(
        default=1, ge=0, description="Total number of consecutive on-time premiums paid"
    )
    sourcing_channel: str = Field(
        default="A", description="Customer acquisition channel (A, B, C, D, or E)"
    )
    residence_area_type: str = Field(
        default="Urban", description="Residence classification (Urban or Rural)"
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "id": "POL-98231",
                "perc_premium_paid_by_cash_credit": 0.42,
                "age": 42,
                "Income": 210000.0,
                "Count_3-6_months_late": 1.0,
                "Count_6-12_months_late": 0.0,
                "Count_more_than_12_months_late": 0.0,
                "application_underwriting_score": 98.6,
                "no_of_premiums_paid": 14,
                "sourcing_channel": "C",
                "residence_area_type": "Urban"
            }
        }


class PredictionResult(BaseModel):
    customer_id: Optional[str] = Field(default="POL-00001", description="Customer identifier")
    default_probability: float
    on_time_probability: float
    predicted_status: str
    risk_tier: str
    recommended_action: str
    intervention_cost: int


@app.get("/", tags=["General"])
def root():
    """Service metadata and API health information."""
    return {
        "service": "Insurance Premium Default Risk Profiler",
        "status": "online",
        "docs_url": "/docs",
        "champion_model": pipeline.model_path.name if pipeline else None,
        "optimal_threshold": pipeline.optimal_threshold if pipeline else None
    }


@app.get("/health", tags=["Monitoring"])
def health():
    """Health check endpoint for container and orchestrator liveness probes."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="InferencePipeline not initialized")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": str(pipeline.model_path.resolve()),
        "optimal_threshold": pipeline.optimal_threshold
    }


@app.post("/predict", response_model=Union[PredictionResult, List[PredictionResult]], tags=["Inference"])
def predict(payload: Union[PolicyholderInput, List[PolicyholderInput]]):
    """Score single or batch customer records for default risk and operational intervention."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference model is unavailable")

    is_single = isinstance(payload, PolicyholderInput)
    records = [payload.model_dump(by_alias=True)] if is_single else [p.model_dump(by_alias=True) for p in payload]

    df_input = pd.DataFrame(records)

    try:
        results_df = pipeline.predict(df_input)
        records_out = results_df.to_dict(orient="records")
        return records_out[0] if is_single else records_out
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("predict:app", host="0.0.0.0", port=8000, reload=True)