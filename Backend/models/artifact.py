from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class MapArtifactMetadata(BaseModel):
    bbox: list[float]
    variable: str
    units: str
    colorbar: dict[str, float | None]
    source_handles: list[str] = Field(default_factory=list)

    @field_validator("bbox")
    @classmethod
    def _bbox_has_four_coordinates(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 coordinates: [min_lon, min_lat, max_lon, max_lat]")
        return v

    @field_validator("colorbar")
    @classmethod
    def _colorbar_has_vmin_vmax(cls, v: dict[str, float | None]) -> dict[str, float | None]:
        if "vmin" not in v or "vmax" not in v:
            raise ValueError("colorbar must include both 'vmin' and 'vmax'")
        return v


class ComparisonPanelRef(BaseModel):
    handle: str
    title: str | None = None


class ComparisonArtifactMetadata(BaseModel):
    mode: Literal["n-panel", "difference"]
    panels: list[ComparisonPanelRef]
    source_handles: list[str] = Field(default_factory=list)

    @field_validator("panels")
    @classmethod
    def _at_least_two_panels(cls, v: list[ComparisonPanelRef]) -> list[ComparisonPanelRef]:
        if len(v) < 2:
            raise ValueError("comparison artifacts require at least 2 panels")
        return v


class TimeseriesSeriesRef(BaseModel):
    label: str
    source_kind: Literal["satellite", "ground"]
    station_id: str | None = None


class TimeseriesArtifactMetadata(BaseModel):
    series: list[TimeseriesSeriesRef]
    source_handles: list[str] = Field(default_factory=list)
    stats: dict[str, Any] | None = None
    coverage: dict[str, Any] | None = None
    exceedance_dates: list[str] | None = None

    @field_validator("series")
    @classmethod
    def _at_least_one_series(cls, v: list[TimeseriesSeriesRef]) -> list[TimeseriesSeriesRef]:
        if len(v) < 1:
            raise ValueError("timeseries artifacts require at least 1 series")
        return v


class ArtifactReference(BaseModel):
    id: str
    type: str
    title: str | None = None
    row_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TableArtifactPayload(BaseModel):
    type: Literal["table"] = "table"
    title: str
    columns: list[str]
    rows: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)
