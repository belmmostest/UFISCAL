"""Pydantic models shared across HTTP and MCP entry points."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from dgce_model.orchestration.policy_package import (
    DEFAULT_COMPREHENSIVE_PARAMS,
    DEFAULT_QUICK_PARAMS,
    DEFAULT_SECTOR_PARAMS,
)


class QuickParams(BaseModel):
    standard_rate: float = Field(
        DEFAULT_QUICK_PARAMS["standard_rate"], ge=0.0, le=0.5
    )
    threshold: float = Field(DEFAULT_QUICK_PARAMS["threshold"], ge=0.0)
    small_biz_threshold: float = Field(
        DEFAULT_QUICK_PARAMS["small_biz_threshold"], ge=1_000_000
    )
    oil_gas_rate: float = Field(DEFAULT_QUICK_PARAMS["oil_gas_rate"], ge=0.0, le=0.8)
    fz_qualifying_rate: float = Field(
        DEFAULT_QUICK_PARAMS["fz_qualifying_rate"], ge=0.0, le=1.0
    )
    sme_election_rate: float = Field(
        DEFAULT_QUICK_PARAMS["sme_election_rate"], ge=0.0, le=1.0
    )
    compliance_rate: float = Field(
        DEFAULT_QUICK_PARAMS["compliance_rate"], ge=0.1, le=1.0
    )
    vat_rate: float = Field(DEFAULT_QUICK_PARAMS["vat_rate"], ge=0.0, le=0.3)
    government_spending_rel_change: float = Field(
        DEFAULT_QUICK_PARAMS["government_spending_rel_change"]
    )
    years: int = Field(DEFAULT_QUICK_PARAMS["years"], ge=1)
    incentives: Dict[str, Any] = Field(default_factory=dict)


class ComprehensiveParams(BaseModel):
    standard_rate: float = Field(
        DEFAULT_COMPREHENSIVE_PARAMS["standard_rate"], ge=0.0, le=0.5
    )
    compliance: float = Field(
        DEFAULT_COMPREHENSIVE_PARAMS["compliance"], ge=0.1, le=1.0
    )
    threshold: float = Field(DEFAULT_COMPREHENSIVE_PARAMS["threshold"], ge=0.0)
    small_biz_threshold: float = Field(
        DEFAULT_COMPREHENSIVE_PARAMS["small_biz_threshold"], ge=1_000_000
    )
    oil_gas_rate: float = Field(
        DEFAULT_COMPREHENSIVE_PARAMS["oil_gas_rate"], ge=0.0, le=0.8
    )
    vat_rate: float = Field(DEFAULT_COMPREHENSIVE_PARAMS["vat_rate"], ge=0.0, le=0.3)
    government_spending_rel_change: float = Field(
        DEFAULT_COMPREHENSIVE_PARAMS["government_spending_rel_change"]
    )
    time_horizon: int = Field(DEFAULT_COMPREHENSIVE_PARAMS["time_horizon"], ge=1)
    incentives: Dict[str, Any] = Field(default_factory=dict)


class SectorParams(BaseModel):
    standard_rate: float = Field(DEFAULT_SECTOR_PARAMS["standard_rate"], ge=0.0, le=0.5)
    compliance_rate: float = Field(
        DEFAULT_SECTOR_PARAMS["compliance_rate"], ge=0.1, le=1.0
    )
    threshold: float = Field(DEFAULT_SECTOR_PARAMS["threshold"], ge=0.0)
    small_biz_threshold: float = Field(
        DEFAULT_SECTOR_PARAMS["small_biz_threshold"], ge=1_000_000
    )
    oil_gas_rate: float = Field(DEFAULT_SECTOR_PARAMS["oil_gas_rate"], ge=0.0, le=0.8)
    incentives: Dict[str, Any] = Field(default_factory=dict)


class SectorRequest(BaseModel):
    sector: str = Field("Manufacturing")
    params: Optional[SectorParams] = None


class StrategicScenario(BaseModel):
    id: str
    overrides: Optional[Dict[str, Any]] = None


class StrategicConfig(BaseModel):
    scenarios: List[StrategicScenario] = Field(default_factory=list)



class PolicyPackageModel(BaseModel):
    quick_params: QuickParams = Field(default_factory=QuickParams)
    comprehensive_params: ComprehensiveParams = Field(default_factory=ComprehensiveParams)
    sector_params: List[SectorRequest] = Field(default_factory=list)
    strategic: StrategicConfig = Field(default_factory=StrategicConfig)

    @validator("sector_params", pre=True, always=True)
    def default_sector(cls, value):
        if not value:
            return [SectorRequest()]  # type: ignore[list-item]
        return value


__all__ = [
    "PolicyPackageModel",
    "QuickParams",
    "ComprehensiveParams",
    "SectorParams",
    "SectorRequest",
    "StrategicScenario",
    "StrategicConfig",
]
