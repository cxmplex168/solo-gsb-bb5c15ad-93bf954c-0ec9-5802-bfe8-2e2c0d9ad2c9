"""FastAPI application exposing the GS1 element-string parser."""

from __future__ import annotations

import json
import re
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from .gs1 import GS1ParseError, parse
from .pickplan import DemandValidationError, pick_plan
from .recall import ItemConflictError, RecallRuleError, recall_impact
from .summary import WeightSummaryError, summarize

app = FastAPI(
    title="GS1 Element String Parser",
    version="1.7.0",
    description=(
        "Parses GS1 input into structured, ordered AI elements. Accepts raw "
        "scan strings (FNC1 / ASCII 29 separated) and GS1 Digital Link URIs "
        "(http/https, AI/value pairs in path and query string). Supports AI "
        "01 (GTIN), 10 (batch/lot), 17 (expiry date), 21 (serial number) and "
        "the weight family 310n (net weight, kg) / 320n (net weight, lb), "
        "where n is 0-9 implied decimal places followed by a fixed 6-digit "
        "value; weight elements additionally carry decimal_value and unit. "
        "POST /weight-summary aggregates the net weights of many scans, "
        "grouped by GTIN and batch, using exact Decimal arithmetic. "
        "POST /weight-pick-plan aggregates scans by GTIN, batch and expiry "
        "and fulfils per-GTIN weight demands in FEFO order (earliest AI 17 "
        "expiry first), allowing a partial final allocation. Each request "
        "carries a planning date as_of and every demand a non-negative "
        "min_remaining_days (default 0): only stock whose actual expiry "
        "date is on or after as_of + min_remaining_days may serve that "
        "demand, and the weight ruled out by that threshold is reported "
        "per demand as excluded_weight. Every demand may instead set "
        'package_mode "whole" to treat each eligible scan as an indivisible '
        "package: the plan then picks the package subset reaching the "
        "demanded quantity with the smallest overage (reported as overage; "
        "shortage never goes negative), or every eligible package when no "
        "subset suffices. "
        "POST /recall-impact inventories a recall scope: scans carry AI 01, "
        "17, 21, one weight AI and an optional AI 10, and the (GTIN, serial) "
        "pair identifies one physical item, so identical re-scans count "
        "once while conflicting batch/expiry/weight scans fail with 422 "
        "naming both indices. Items are matched against ordered rules by "
        "GTIN, an optional batch set and an optional closed expiry-date "
        "interval; each item belongs only to the first matching rule, and "
        "the response reports per-rule package counts and six-place "
        "half-even totals plus an unmatched summary."
    ),
)


class ParseRequest(BaseModel):
    data: str = Field(
        ...,
        title="Raw scan string or GS1 Digital Link URI",
        description=(
            "Either a raw GS1 element string as scanned (variable-length "
            "fields separated by ASCII 29, encoded in JSON as , or a "
            "GS1 Digital Link URI beginning with http:// or https://. "
            "Digital Link values are percent-decoded as UTF-8; path segments "
            "must form AI/value pairs; each AI may appear only once across "
            "path and query string."
        ),
        examples=[
            "01095011015300031725013110ABC12321SN0001",
            "https://id.gs1.org/01/09501101530003/17/250131?10=ABC123&21=SN0001",
        ],
    )


class Element(BaseModel):
    ai: str = Field(description="Application identifier, e.g. '01'")
    name: str = Field(description="Short element name, e.g. 'GTIN'")
    value: str = Field(
        description="Field value without AI and separators (raw digits for "
        "310n/320n, before applying the implied decimal point)"
    )
    offset: int = Field(description="Byte offset of the AI within the input")
    decimal_value: float | None = Field(
        default=None,
        description=(
            "310n/320n only: the 6-digit value interpreted with n implied "
            "decimal places. Absent for all other AIs."
        ),
    )
    unit: str | None = Field(
        default=None,
        description='310n/320n only: measurement unit, "kg" (310n) or "lb" (320n). '
        "Absent for all other AIs.",
    )


class ParseResponse(BaseModel):
    elements: list[Element]


class WeightSummaryRequest(BaseModel):
    scans: list[str] = Field(
        ...,
        description=(
            "Raw GS1 scan strings and/or GS1 Digital Link URIs, one per item. "
            "Each must contain exactly one AI 01, at most one AI 10 and "
            "exactly one weight AI 310n/320n."
        ),
        examples=[
            [
                "010950110153000310B13103002513",
                "010950110153000310B13202002513",
            ]
        ],
    )
    target_unit: Literal["kg", "lb"] = Field(
        default="kg",
        description='Unit the group totals are converted to: "kg" or "lb" (default "kg").',
    )


class WeightGroup(BaseModel):
    """OpenAPI shape of one group; totals are rendered as exact JSON numbers."""

    gtin: str = Field(description="AI 01 GTIN-14 shared by the group")
    batch: str | None = Field(
        description="AI 10 batch/lot; null for scans that carried no batch"
    )
    count: int = Field(description="Number of scans aggregated into the group")
    unit: Literal["kg", "lb"] = Field(description="Unit of total (the target unit)")
    total: float = Field(
        description=(
            "Sum of net weights converted to the target unit, rounded to six "
            "decimal places with round-half-even. Rendered as a JSON number "
            "(e.g. 2.513000); computed internally with Decimal, never float."
        )
    )


class WeightSummaryResponse(BaseModel):
    target_unit: Literal["kg", "lb"]
    groups: list[WeightGroup]


class Demand(BaseModel):
    gtin: str = Field(
        ...,
        min_length=1,
        description="GTIN (AI 01 value) to satisfy; must be unique within one request.",
        examples=["09501101530003"],
    )
    quantity: Decimal = Field(
        ...,
        gt=0,
        description="Required net weight, expressed in target_unit.",
        examples=[3.5],
    )
    min_remaining_days: int = Field(
        default=0,
        description=(
            "Minimum remaining shelf life in days, counted from as_of. Only "
            "stock whose actual AI 17 expiry date is on or after "
            "as_of + min_remaining_days may serve this demand; the boundary "
            "date itself is eligible. Must be a non-negative integer; a "
            "negative value is rejected with 422 naming the demand_index."
        ),
        examples=[30],
    )
    package_mode: Literal["splittable", "whole"] = Field(
        default="splittable",
        description=(
            '"splittable" (default) allocates weight freely, allowing a '
            'partial final allocation. "whole" treats every eligible scan as '
            "an indivisible package: the plan picks the package subset "
            "reaching quantity with the smallest overage (ties broken by the "
            "lexicographically smallest FEFO index sequence), or every "
            "eligible package when no subset reaches quantity. A demand with "
            "more than 30 eligible packages is rejected with 422 naming the "
            "demand_index."
        ),
        examples=["whole"],
    )


class WeightPickPlanRequest(BaseModel):
    scans: list[str] = Field(
        ...,
        description=(
            "Raw GS1 scan strings and/or GS1 Digital Link URIs, one per item. "
            "Each must contain exactly one AI 01, exactly one AI 17 (expiry), "
            "at most one AI 10 and exactly one weight AI 310n/320n."
        ),
        examples=[
            [
                "01095011015300031725013110B1\x1d3103002000",
                "01095011015300031725022810B2\x1d3103002000",
            ]
        ],
    )
    target_unit: Literal["kg", "lb"] = Field(
        default="kg",
        description=(
            'Unit of the demand quantities and of every weight in the response: '
            '"kg" or "lb" (default "kg").'
        ),
    )
    as_of: date = Field(
        ...,
        title="Planning reference date",
        description=(
            "Planning date in YYYY-MM-DD form; must be a real calendar date "
            "(invalid values are rejected with 422). Per demand, stock is "
            "eligible only when its actual AI 17 expiry date -- day 00 "
            "meaning the last day of the month -- is on or after "
            "as_of + min_remaining_days."
        ),
        examples=["2025-01-15"],
    )
    demands: list[Demand] = Field(
        ...,
        description=(
            "One demand per GTIN; quantity is expressed in target_unit. "
            "A GTIN repeated across demands is rejected with 422."
        ),
        examples=[[{"gtin": "09501101530003", "quantity": 3}]],
    )

    @field_validator("as_of", mode="before")
    @classmethod
    def _as_of_is_yyyy_mm_dd(cls, value: Any) -> Any:
        """Accept only a YYYY-MM-DD string (or an already-parsed date)."""
        if isinstance(value, date):
            return value
        if not isinstance(value, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", value
        ):
            raise ValueError("as_of must be a date string in YYYY-MM-DD format")
        return value


class Allocation(BaseModel):
    """OpenAPI shape of one allocation; weight is rendered as an exact JSON number."""

    batch: str | None = Field(
        description="AI 10 batch/lot of the source group; null for scans without a batch"
    )
    expiry: str = Field(
        description="AI 17 value (YYMMDD) of the source group, exactly as scanned"
    )
    weight: float = Field(
        description=(
            "Net weight taken from this group, in the target unit, rounded to "
            "six decimal places with round-half-even. Computed internally with "
            "Decimal, never float."
        )
    )


class DemandPlan(BaseModel):
    """OpenAPI shape of one fulfilled demand; weights render as exact JSON numbers."""

    gtin: str = Field(description="GTIN of the demand")
    quantity: float = Field(
        description="Requested quantity, echoed in the target unit (six decimal places)"
    )
    fulfilled: float = Field(
        description="Total weight allocated to the demand (six decimal places)"
    )
    shortage: float = Field(
        description="quantity - fulfilled; 0.000000 when fully satisfied (six decimal places)"
    )
    overage: float = Field(
        description=(
            "fulfilled - quantity when a whole-package plan overshoots the "
            "demand; always 0.000000 for splittable demands (six decimal places)"
        )
    )
    excluded_weight: float = Field(
        description=(
            "Total weight of this GTIN's stock ruled out by the "
            "as_of + min_remaining_days threshold, in the target unit "
            "(six decimal places)"
        )
    )
    allocations: list[Allocation]


class WeightPickPlanResponse(BaseModel):
    target_unit: Literal["kg", "lb"]
    demands: list[DemandPlan]


class RecallRule(BaseModel):
    gtin: str = Field(
        ...,
        min_length=1,
        description=(
            "GTIN (AI 01 value) whose items this rule covers. Rules are "
            "evaluated in request order and each item belongs to the first "
            "matching rule only."
        ),
        examples=["09501101530003"],
    )
    batches: list[str] | None = Field(
        default=None,
        description=(
            "Optional set of AI 10 batch/lot values (as a JSON array; "
            "duplicates are harmless) restricting the rule to those "
            "batches. Omit or null to match every batch of the GTIN, "
            "including items without a batch; an item without a batch "
            "never matches a rule that names batches. An empty array is "
            "rejected with 422 naming the rule_index."
        ),
        examples=[["B1", "B2"]],
    )
    expiry_from: date | None = Field(
        default=None,
        title="Closed interval lower bound",
        description=(
            "Optional earliest actual AI 17 expiry date (YYYY-MM-DD; "
            "day 00 means the last day of the month) that is still inside "
            "the recall; the boundary date itself matches. Must not be "
            "later than expiry_to."
        ),
        examples=["2025-01-01"],
    )
    expiry_to: date | None = Field(
        default=None,
        title="Closed interval upper bound",
        description=(
            "Optional latest actual AI 17 expiry date (YYYY-MM-DD) still "
            "inside the recall; the boundary date itself matches. Must not "
            "be earlier than expiry_from."
        ),
        examples=["2025-06-30"],
    )

    @field_validator("expiry_from", "expiry_to", mode="before")
    @classmethod
    def _expiry_is_yyyy_mm_dd(cls, value: Any) -> Any:
        """Accept only a YYYY-MM-DD string (or an already-parsed date)."""
        if value is None or isinstance(value, date):
            return value
        if not isinstance(value, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", value
        ):
            raise ValueError("expiry dates must be strings in YYYY-MM-DD format")
        return value

    @field_validator("batches")
    @classmethod
    def _batch_values_non_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is not None:
            for i, batch in enumerate(value):
                if not isinstance(batch, str) or not batch:
                    raise ValueError(
                        f"batches[{i}] must be a non-empty AI 10 value"
                    )
        return value


class RecallImpactRequest(BaseModel):
    scans: list[str] = Field(
        ...,
        description=(
            "Raw GS1 scan strings and/or GS1 Digital Link URIs, one per "
            "item. Each must contain exactly one AI 01, exactly one AI 17 "
            "(expiry), exactly one AI 21 (serial), at most one AI 10 "
            "(batch) and exactly one weight AI 310n/320n."
        ),
        examples=[
            [
                "01095011015300031725013110B1\x1d21SN0001\x1d3103002000",
                "01095011015300031725033110B2\x1d21SN0002\x1d3103003000",
            ]
        ],
    )
    target_unit: Literal["kg", "lb"] = Field(
        default="kg",
        description=(
            'Unit of every weight in the response: "kg" or "lb" (default "kg").'
        ),
    )
    rules: list[RecallRule] = Field(
        ...,
        description=(
            "Ordered recall rules; an item is assigned to the first rule "
            "whose GTIN, optional batch set and optional closed expiry "
            "interval all match."
        ),
        examples=[[{"gtin": "09501101530003", "batches": ["B1"]}]],
    )


class RecallRuleResult(BaseModel):
    """OpenAPI shape of one rule's hit; the weight renders as an exact number."""

    gtin: str = Field(description="GTIN of the rule, echoed")
    batches: list[str] | None = Field(
        description="Rule batch set echoed and deduplicated/sorted, or null"
    )
    expiry_from: date | None = Field(
        description="Closed interval lower bound echoed (YYYY-MM-DD), or null"
    )
    expiry_to: date | None = Field(
        description="Closed interval upper bound echoed (YYYY-MM-DD), or null"
    )
    package_count: int = Field(
        description="Number of distinct physical items assigned to this rule"
    )
    total_weight: float = Field(
        description=(
            "Sum of the matched items' net weights in the target unit, "
            "rounded to six decimal places with round-half-even. Computed "
            "internally with Decimal, never float."
        )
    )


class RecallUnmatched(BaseModel):
    package_count: int = Field(
        description="Number of distinct physical items no rule matched"
    )
    total_weight: float = Field(
        description=(
            "Total net weight of the unmatched items in the target unit, "
            "six decimal places with round-half-even."
        )
    )


class RecallImpactResponse(BaseModel):
    target_unit: Literal["kg", "lb"]
    rules: list[RecallRuleResult]
    unmatched: RecallUnmatched


SIX_PLACES = Decimal("0.000001")


class _RawNumber(str):
    """Fixed-point numeric text, e.g. '2.513000', emitted verbatim as JSON."""


def _dump(value: Any) -> str:
    """JSON serializer that splices :class:`_RawNumber` tokens raw.

    The stdlib encoder stringifies Decimal and FastAPI's jsonable_encoder
    turns it into a float; totals must instead appear as exact six-place
    JSON numbers such as ``2.513000``, so they are pre-formatted and
    spliced in here as raw tokens.
    """
    if isinstance(value, _RawNumber):
        return str(value)
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(_dump(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                f"{json.dumps(str(k), ensure_ascii=False)}:{_dump(v)}"
                for k, v in value.items()
            )
            + "}"
        )
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _six(value: Decimal) -> _RawNumber:
    """Fixed-point token: ``value`` quantized half-even to six places."""
    return _RawNumber(
        format(value.quantize(SIX_PLACES, rounding=ROUND_HALF_EVEN), "f")
    )


def _build_summary_response(result: dict) -> Response:
    """Quantize totals half-even to six places and render exact JSON."""
    payload = {
        "target_unit": result["target_unit"],
        "groups": [
            {
                "gtin": group["gtin"],
                "batch": group["batch"],
                "count": group["count"],
                "unit": group["unit"],
                "total": _six(group["total"]),
            }
            for group in result["groups"]
        ],
    }
    return Response(content=_dump(payload), media_type="application/json")


def _build_pick_plan_response(result: dict) -> Response:
    """Quantize every weight half-even to six places and render exact JSON."""
    payload = {
        "target_unit": result["target_unit"],
        "demands": [
            {
                "gtin": demand["gtin"],
                "quantity": _six(demand["quantity"]),
                "fulfilled": _six(demand["fulfilled"]),
                "shortage": _six(demand["shortage"]),
                "overage": _six(demand["overage"]),
                "excluded_weight": _six(demand["excluded_weight"]),
                "allocations": [
                    {
                        "batch": allocation["batch"],
                        "expiry": allocation["expiry"],
                        "weight": _six(allocation["weight"]),
                    }
                    for allocation in demand["allocations"]
                ],
            }
            for demand in result["demands"]
        ],
    }
    return Response(content=_dump(payload), media_type="application/json")


def _build_recall_impact_response(result: dict) -> Response:
    """Quantize totals half-even to six places and render exact JSON."""
    payload = {
        "target_unit": result["target_unit"],
        "rules": [
            {
                "gtin": rule["gtin"],
                "batches": rule["batches"],
                "expiry_from": (
                    rule["expiry_from"].isoformat()
                    if rule["expiry_from"] is not None
                    else None
                ),
                "expiry_to": (
                    rule["expiry_to"].isoformat()
                    if rule["expiry_to"] is not None
                    else None
                ),
                "package_count": rule["package_count"],
                "total_weight": _six(rule["total_weight"]),
            }
            for rule in result["rules"]
        ],
        "unmatched": {
            "package_count": result["unmatched"]["package_count"],
            "total_weight": _six(result["unmatched"]["total_weight"]),
        },
    }
    return Response(content=_dump(payload), media_type="application/json")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/parse", response_model=ParseResponse, response_model_exclude_none=True)
def parse_scan(req: ParseRequest) -> ParseResponse:
    try:
        elements = parse(req.data)
    except GS1ParseError as exc:
        raise HTTPException(status_code=422, detail=exc.to_dict()) from exc
    return ParseResponse(elements=[Element(**vars(e)) for e in elements])


@app.post(
    "/weight-summary",
    response_model=WeightSummaryResponse,
)
def weight_summary(req: WeightSummaryRequest) -> Response:
    try:
        result = summarize(req.scans, req.target_unit)
    except WeightSummaryError as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc
    return _build_summary_response(result)


@app.post(
    "/weight-pick-plan",
    response_model=WeightPickPlanResponse,
)
def weight_pick_plan(req: WeightPickPlanRequest) -> Response:
    try:
        result = pick_plan(
            req.scans,
            req.target_unit,
            [
                (d.gtin, d.quantity, d.min_remaining_days, d.package_mode)
                for d in req.demands
            ],
            req.as_of,
        )
    except (WeightSummaryError, DemandValidationError) as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc
    return _build_pick_plan_response(result)


@app.post(
    "/recall-impact",
    response_model=RecallImpactResponse,
)
def recall_impact_endpoint(req: RecallImpactRequest) -> Response:
    try:
        result = recall_impact(
            req.scans,
            req.target_unit,
            [
                (r.gtin, set(r.batches) if r.batches is not None else None,
                 r.expiry_from, r.expiry_to)
                for r in req.rules
            ],
        )
    except (
        WeightSummaryError,
        RecallRuleError,
        ItemConflictError,
    ) as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc
    return _build_recall_impact_response(result)
