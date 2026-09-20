"""Recall-scope inventory over parsed GS1 scans.

Each scan is fed through :func:`app.gs1.parse` and must then contain:

* exactly one AI 01 (GTIN),
* exactly one AI 17 (expiry date; day ``00`` denotes the last day of the
  month, per the GS1 General Specifications),
* exactly one AI 21 (serial number),
* exactly one weight element of the 310n (kg) / 320n (lb) family,
* zero or one AI 10 (batch/lot) -- optional.

A physical item is identified by the pair **(GTIN, serial number)**.
Repeated scans of the same pair are therefore scans of one item:

* when batch, expiry and weight all match the first scan exactly the
  duplicate counts once;
* when any of them differs the whole request fails with
  :class:`ItemConflictError` carrying both the current scan index and the
  index of the first scan of that item.

The de-duplicated items are then matched against an ordered list of
recall rules.  A rule matches an item when:

1. the item's GTIN equals the rule GTIN;
2. the rule has no ``batches`` set, or the item's batch is a member of it
   (items without a batch only match rules without a batch set);
3. the rule has no expiry interval, or the item's *actual* expiry date
   (AI 17 day ``00`` -> last day of the month) lies in the closed
   interval ``[expiry_from, expiry_to]`` (one of the bounds may be
   omitted).

Each item is assigned to the **first** matching rule in request order;
items no rule matches are reported as unmatched.  Every weight is
rebuilt with :class:`decimal.Decimal` from the original six digits and
converted with the fixed factors of :mod:`app.summary`; the response
per rule gives the package count and the half-even six-place total
weight, in rule order, plus an unmatched summary.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, localcontext

from .gs1 import GS1ParseError, Element, decode_weight, parse
from .pickplan import _expiry_date
from .summary import (
    ARITHMETIC_PRECISION,
    ScanValidationError,
    WeightSummaryError,
    _failure,
    _to_target,
)

REQUIRED_AI_TEXT = (
    "exactly one AI 01 (GTIN), exactly one AI 17 (expiry date), "
    "exactly one AI 21 (serial number), at most one AI 10 (batch/lot), "
    "and exactly one weight AI 310n/320n"
)


class RecallRuleError(Exception):
    """A recall rule violated the request shape.

    ``detail`` is the complete 422 payload: ``rule_index`` plus the
    validation error under ``error``, mirroring the demand-level shape
    used by ``/weight-pick-plan``.
    """

    def __init__(self, detail: dict):
        super().__init__(detail["error"]["message"])
        self.detail = detail


class ItemConflictError(Exception):
    """Two scans of the same (GTIN, serial) item carry conflicting data.

    ``detail`` is the complete 422 payload: the current ``scan_index``,
    the ``first_scan_index`` at which the item was first seen and the
    conflict description under ``error``.
    """

    def __init__(self, detail: dict):
        super().__init__(detail["error"]["message"])
        self.detail = detail


def _rule_failure(rule_index: int, code: str, message: str, expected: str, found: str):
    """Wrap a rule validation error with the offending rule index."""
    return RecallRuleError(
        {
            "rule_index": rule_index,
            "error": {
                "error": code,
                "message": message,
                "expected": expected,
                "found": found,
            },
        }
    )


def _conflict(
    index: int,
    first_index: int,
    field: str,
    current: str,
    first: str,
    gtin: str,
    serial: str,
) -> ItemConflictError:
    """Build the 422 payload for an attribute clash between repeated scans."""
    return ItemConflictError(
        {
            "scan_index": index,
            "first_scan_index": first_index,
            "error": {
                "error": "CONFLICTING_ITEM_ATTRIBUTES",
                "message": (
                    f"scan {index}: {field} of GTIN '{gtin}' serial '{serial}' "
                    f"is {current!r} but scan {first_index} first recorded it "
                    f"as {first!r}; repeated scans of one (GTIN, serial) item "
                    "must carry identical batch, expiry and weight"
                ),
                "expected": f"{field} {first!r} as first scanned at index {first_index}",
                "found": current,
            },
        }
    )


def _extract(
    index: int, elements: list[Element]
) -> tuple[str, str | None, str, str, Decimal, str]:
    """Pull (GTIN, batch or None, AI 17 value, serial, exact weight, unit)."""
    gtins = [e for e in elements if e.ai == "01"]
    expiries = [e for e in elements if e.ai == "17"]
    serials = [e for e in elements if e.ai == "21"]
    batches = [e for e in elements if e.ai == "10"]
    weights = [e for e in elements if e.unit is not None]

    if len(gtins) != 1:
        if not gtins:
            raise ScanValidationError(
                "MISSING_AI",
                f"scan {index}: missing required AI 01 (GTIN)",
                expected=REQUIRED_AI_TEXT,
            )
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"scan {index}: AI 01 (GTIN) appears {len(gtins)} times; exactly one is required",
            expected="exactly one AI 01 (GTIN)",
            found=f"{len(gtins)} AI 01 elements",
            offset=gtins[1].offset,
        )

    if len(expiries) != 1:
        if not expiries:
            raise ScanValidationError(
                "MISSING_AI",
                f"scan {index}: missing required AI 17 (expiry date)",
                expected=REQUIRED_AI_TEXT,
            )
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"scan {index}: AI 17 (expiry date) appears {len(expiries)} times; exactly one is required",
            expected="exactly one AI 17 (expiry date)",
            found=f"{len(expiries)} AI 17 elements",
            offset=expiries[1].offset,
        )

    if len(serials) != 1:
        if not serials:
            raise ScanValidationError(
                "MISSING_AI",
                f"scan {index}: missing required AI 21 (serial number)",
                expected=REQUIRED_AI_TEXT,
            )
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"scan {index}: AI 21 (serial number) appears {len(serials)} times; exactly one is required",
            expected="exactly one AI 21 (serial number)",
            found=f"{len(serials)} AI 21 elements",
            offset=serials[1].offset,
        )

    if len(batches) > 1:
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"scan {index}: AI 10 (batch/lot) appears {len(batches)} times; at most one is allowed",
            expected="at most one AI 10 (batch/lot)",
            found=f"{len(batches)} AI 10 elements",
            offset=batches[1].offset,
        )

    if len(weights) != 1:
        if not weights:
            raise ScanValidationError(
                "MISSING_AI",
                f"scan {index}: missing required weight AI 310n/320n",
                expected=REQUIRED_AI_TEXT,
            )
        ais = ", ".join(e.ai for e in weights)
        raise ScanValidationError(
            "DUPLICATE_AI",
            f"scan {index}: {len(weights)} weight elements ({ais}); "
            "exactly one 310n/320n is required",
            expected="exactly one weight AI 310n (kg) or 320n (lb)",
            found=ais,
            offset=weights[1].offset,
        )

    weight = weights[0]
    value = decode_weight(weight.value, int(weight.ai[3]))
    return (
        gtins[0].value,
        batches[0].value if batches else None,
        expiries[0].value,
        serials[0].value,
        value,
        weight.unit,
    )


def _check_rules(rules: list[tuple]) -> None:
    """Reject empty batch sets and inverted expiry intervals."""
    for rule_index, (
        _gtin,
        batches,
        expiry_from,
        expiry_to,
    ) in enumerate(rules):
        if batches is not None and len(batches) == 0:
            raise _rule_failure(
                rule_index,
                "INVALID_BATCHES",
                f"rule {rule_index}: batches must be a non-empty set or omitted; "
                "an empty list matches nothing",
                expected="non-empty batches list, or the field omitted",
                found="[]",
            )
        if expiry_from is not None and expiry_to is not None:
            if expiry_from > expiry_to:
                raise _rule_failure(
                    rule_index,
                    "INVALID_EXPIRY_RANGE",
                    f"rule {rule_index}: expiry_from {expiry_from.isoformat()} "
                    f"is after expiry_to {expiry_to.isoformat()}",
                    expected="expiry_from on or before expiry_to",
                    found=(
                        f"[{expiry_from.isoformat()}, {expiry_to.isoformat()}]"
                    ),
                )


def _matches(
    item: dict,
    gtin: str,
    batches: frozenset[str] | None,
    lo: date | None,
    hi: date | None,
) -> bool:
    """True when the item falls inside one recall rule's scope."""
    if item["gtin"] != gtin:
        return False
    if batches is not None and item["batch"] not in batches:
        # An item without a batch cannot belong to an explicit batch set.
        return False
    if lo is not None and item["expiry_date"] < lo:
        return False
    if hi is not None and item["expiry_date"] > hi:
        return False
    return True


def recall_impact(scans: list[str], target_unit: str, rules: list[tuple]) -> dict:
    """Parse and de-duplicate scans, then assign each item to the first
    matching recall rule.

    ``rules`` is a list of ``(gtin, batches, expiry_from, expiry_to)``
    tuples where ``batches`` is ``None`` or a set of batch values, and
    the two expiry bounds are ``None`` or real :class:`date` values
    forming a closed interval.

    Returns ``{"target_unit": ..., "rules": [{gtin, batches,
    expiry_from, expiry_to, package_count, total_weight}, ...],
    "unmatched": {"package_count", "total_weight"}}`` in rule order;
    every weight is an unrounded :class:`Decimal` in the target unit.
    """
    normalized = [
        (
            gtin,
            frozenset(batches) if batches is not None else None,
            expiry_from,
            expiry_to,
        )
        for gtin, batches, expiry_from, expiry_to in rules
    ]
    _check_rules(normalized)

    # Physical item keyed by (GTIN, serial); first-seen order is kept.
    items: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []

    with localcontext() as ctx:
        ctx.prec = ARITHMETIC_PRECISION
        for index, scan in enumerate(scans):
            try:
                elements = parse(scan)
            except GS1ParseError as exc:
                raise _failure(index, exc) from exc
            try:
                gtin, batch, expiry, serial, weight, unit = _extract(index, elements)
            except ScanValidationError as exc:
                raise _failure(index, exc) from exc

            converted = _to_target(weight, unit, target_unit)
            key = (gtin, serial)
            if key not in items:
                order.append(key)
                items[key] = {
                    "gtin": gtin,
                    "batch": batch,
                    "expiry": expiry,
                    "expiry_date": _expiry_date(expiry),
                    "serial": serial,
                    "weight": converted,
                    "first_index": index,
                }
                continue

            first = items[key]
            if batch != first["batch"]:
                raise _conflict(
                    index, first["first_index"], "batch",
                    str(batch), str(first["batch"]), gtin, serial,
                )
            if expiry != first["expiry"]:
                raise _conflict(
                    index, first["first_index"], "expiry",
                    expiry, first["expiry"], gtin, serial,
                )
            if converted != first["weight"]:
                raise _conflict(
                    index, first["first_index"], "weight",
                    str(converted), str(first["weight"]), gtin, serial,
                )
            # Identical re-scan of the same physical item: counts once.

        # rule_index -> [package_count, total weight in target unit]
        matched: dict[int, list] = {}
        unmatched_count = 0
        unmatched_weight = Decimal(0)

        for key in order:
            item = items[key]
            for rule_index, (gtin, batches, lo, hi) in enumerate(normalized):
                if _matches(item, gtin, batches, lo, hi):
                    slot = matched.setdefault(rule_index, [0, Decimal(0)])
                    slot[0] += 1
                    slot[1] += item["weight"]
                    break
            else:
                unmatched_count += 1
                unmatched_weight += item["weight"]

    rule_results = []
    for rule_index, (gtin, batches, lo, hi) in enumerate(normalized):
        count, total = matched.get(rule_index, [0, Decimal(0)])
        rule_results.append(
            {
                "gtin": gtin,
                "batches": sorted(batches) if batches is not None else None,
                "expiry_from": lo,
                "expiry_to": hi,
                "package_count": count,
                "total_weight": total,
            }
        )

    return {
        "target_unit": target_unit,
        "rules": rule_results,
        "unmatched": {
            "package_count": unmatched_count,
            "total_weight": unmatched_weight,
        },
    }
