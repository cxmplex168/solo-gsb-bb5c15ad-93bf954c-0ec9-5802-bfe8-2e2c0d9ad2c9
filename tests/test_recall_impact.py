"""End-to-end tests for the /recall-impact endpoint."""

from decimal import Decimal

from fastapi.testclient import TestClient

from app.main import app
from app.summary import KG_TO_LB

client = TestClient(app)

GS = "\x1d"
GTIN_OK = "09501101530003"  # valid Mod-10 check digit
GTIN2 = "00000000000017"  # second valid GTIN


def impact(scans, rules, target_unit=None):
    body = {"scans": scans, "rules": rules}
    if target_unit is not None:
        body["target_unit"] = target_unit
    return client.post("/recall-impact", json=body)


def scan(gtin=GTIN_OK, expiry="250131", serial="SN1", batch=None,
         weight="3103002000"):
    """Build an element-string scan with one 01/17/21/weight, optional 10.

    The default weight 3103 002000 is exactly 2 kg.
    """
    parts = [f"01{gtin}", f"17{expiry}"]
    if batch is not None:
        parts.append(f"10{batch}{GS}")
    parts.append(f"21{serial}{GS}")
    parts.append(weight)
    return "".join(parts)


def rule(gtin=GTIN_OK, batches=None, expiry_from=None, expiry_to=None):
    r = {"gtin": gtin}
    if batches is not None:
        r["batches"] = batches
    if expiry_from is not None:
        r["expiry_from"] = expiry_from
    if expiry_to is not None:
        r["expiry_to"] = expiry_to
    return r


# ------------------------------------------------------------- rule matching


def test_items_grouped_by_gtin_rules_in_rule_order():
    resp = impact(
        [
            scan(serial="A"),
            scan(serial="B"),
            scan(gtin=GTIN2, serial="C"),
        ],
        [rule(gtin=GTIN2), rule()],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["target_unit"] == "kg"
    first, second = body["rules"]
    assert first["gtin"] == GTIN2
    assert first["package_count"] == 1
    assert first["total_weight"] == 2.0
    assert second["gtin"] == GTIN_OK
    assert second["package_count"] == 2
    assert second["total_weight"] == 4.0
    assert body["unmatched"] == {"package_count": 0, "total_weight": 0.0}


def test_rules_keep_request_order_including_zero_hits():
    resp = impact([scan(serial="A")], [rule(), rule(gtin=GTIN2)])
    rules = resp.json()["rules"]
    assert [r["gtin"] for r in rules] == [GTIN_OK, GTIN2]
    assert [r["package_count"] for r in rules] == [1, 0]


def test_empty_scans_zero_everything():
    resp = impact([], [rule(), rule(gtin=GTIN2)])
    assert resp.status_code == 200
    body = resp.json()
    assert [r["package_count"] for r in body["rules"]] == [0, 0]
    assert all(r["total_weight"] == 0.0 for r in body["rules"])
    assert body["unmatched"] == {"package_count": 0, "total_weight": 0.0}
    assert '"total_weight":0.000000' in resp.text


def test_empty_rules_put_everything_in_unmatched():
    resp = impact(
        [scan(serial="A"), scan(serial="B", weight="3103003000")],
        [],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["rules"] == []
    assert body["unmatched"] == {"package_count": 2, "total_weight": 5.0}


def test_unmatched_gtin_is_summary_only():
    resp = impact(
        [scan(serial="A"), scan(gtin=GTIN2, serial="C")],
        [rule()],
    )
    body = resp.json()
    assert body["rules"][0]["package_count"] == 1
    assert body["unmatched"] == {"package_count": 1, "total_weight": 2.0}


def test_batch_set_matches_only_listed_batches():
    resp = impact(
        [
            scan(serial="A", batch="B1"),
            scan(serial="B", batch="B2"),
            scan(serial="C", batch="B3"),
        ],
        [rule(batches=["B1", "B3"])],
    )
    r = resp.json()["rules"][0]
    assert r["package_count"] == 2
    assert r["total_weight"] == 4.0
    assert resp.json()["unmatched"]["package_count"] == 1


def test_item_without_batch_never_matches_named_batch_rule():
    resp = impact(
        [scan(serial="A"), scan(serial="B", batch="B1")],
        [rule(batches=["B1"])],
    )
    body = resp.json()
    assert body["rules"][0]["package_count"] == 1
    assert body["unmatched"]["package_count"] == 1
    assert body["unmatched"]["total_weight"] == 2.0


def test_rule_without_batches_matches_batched_and_unbatched_items():
    resp = impact(
        [scan(serial="A"), scan(serial="B", batch="B1")],
        [rule()],
    )
    r = resp.json()["rules"][0]
    assert r["package_count"] == 2
    assert r["total_weight"] == 4.0


def test_batch_set_is_echoed_deduplicated_and_sorted():
    resp = impact(
        [],
        [rule(batches=["B3", "B1", "B1"])],
    )
    assert resp.json()["rules"][0]["batches"] == ["B1", "B3"]


def test_expiry_closed_interval_is_inclusive_on_both_bounds():
    resp = impact(
        [
            scan(serial="A", expiry="241231"),  # before
            scan(serial="B", expiry="250101"),  # lower bound
            scan(serial="C", expiry="250131"),  # upper bound
            scan(serial="D", expiry="250201"),  # after
        ],
        [rule(expiry_from="2025-01-01", expiry_to="2025-01-31")],
    )
    r = resp.json()["rules"][0]
    assert r["package_count"] == 2
    assert r["total_weight"] == 4.0
    assert resp.json()["unmatched"]["package_count"] == 2


def test_expiry_lower_bound_only():
    resp = impact(
        [scan(serial="A", expiry="250131"), scan(serial="B", expiry="260131")],
        [rule(expiry_from="2026-01-01")],
    )
    assert resp.json()["rules"][0]["package_count"] == 1
    assert resp.json()["unmatched"]["package_count"] == 1


def test_expiry_upper_bound_only():
    resp = impact(
        [scan(serial="A", expiry="250131"), scan(serial="B", expiry="260131")],
        [rule(expiry_to="2025-12-31")],
    )
    assert resp.json()["rules"][0]["package_count"] == 1
    assert resp.json()["unmatched"]["package_count"] == 1


def test_day_00_expiry_compares_as_last_day_of_month():
    # 250200 = 2025-02-28 (2025 is not a leap year): matches the boundary.
    resp = impact(
        [scan(serial="A", expiry="250200")],
        [rule(expiry_from="2025-02-28", expiry_to="2025-02-28")],
    )
    assert resp.json()["rules"][0]["package_count"] == 1


def test_all_rule_conditions_must_hold_together():
    # Right GTIN and in the expiry window, but wrong batch -> unmatched.
    resp = impact(
        [scan(serial="A", batch="OTHER", expiry="250131")],
        [rule(batches=["B1"], expiry_from="2025-01-01", expiry_to="2025-12-31")],
    )
    body = resp.json()
    assert body["rules"][0]["package_count"] == 0
    assert body["unmatched"]["package_count"] == 1


def test_first_matching_rule_wins_with_overlapping_rules():
    resp = impact(
        [scan(serial="A", batch="B1"), scan(serial="B", batch="B2")],
        [rule(), rule(batches=["B1"])],
    )
    first, second = resp.json()["rules"]
    assert first["package_count"] == 2
    assert second["package_count"] == 0


def test_item_falls_through_non_matching_rules_to_a_later_match():
    resp = impact(
        [scan(serial="A", batch="B2")],
        [rule(batches=["B1"]), rule(batches=["B2"]), rule()],
    )
    rules = resp.json()["rules"]
    assert [r["package_count"] for r in rules] == [0, 1, 0]


def test_batch_and_expiry_filters_combine_across_gtins():
    resp = impact(
        [
            scan(serial="A", batch="B1", expiry="250131"),
            scan(gtin=GTIN2, serial="B", batch="B1", expiry="260131"),
        ],
        [
            rule(batches=["B1"], expiry_to="2025-12-31"),
            rule(gtin=GTIN2, batches=["B1"]),
        ],
    )
    first, second = resp.json()["rules"]
    assert first["package_count"] == 1  # the GTIN_OK item only
    assert second["package_count"] == 1
    assert resp.json()["unmatched"]["package_count"] == 0


def test_digital_link_scans_accepted():
    uri = (
        "https://id.gs1.org/01/09501101530003/17/250131/21/SN1/3103/002000?10=B1"
    )
    resp = impact([uri], [rule(batches=["B1"])])
    assert resp.status_code == 200
    r = resp.json()["rules"][0]
    assert r["package_count"] == 1
    assert r["total_weight"] == 2.0


def test_rule_criteria_are_echoed():
    resp = impact(
        [],
        [rule(batches=["B1"], expiry_from="2025-01-01", expiry_to="2025-06-30")],
    )
    r = resp.json()["rules"][0]
    assert r == {
        "gtin": GTIN_OK,
        "batches": ["B1"],
        "expiry_from": "2025-01-01",
        "expiry_to": "2025-06-30",
        "package_count": 0,
        "total_weight": 0.0,
    }


def test_response_shape():
    resp = impact([scan(serial="A")], [rule()])
    body = resp.json()
    assert set(body) == {"target_unit", "rules", "unmatched"}
    assert set(body["rules"][0]) == {
        "gtin",
        "batches",
        "expiry_from",
        "expiry_to",
        "package_count",
        "total_weight",
    }
    assert set(body["unmatched"]) == {"package_count", "total_weight"}


# ------------------------------------------------------------- de-duplication


def test_identical_rescan_of_one_item_counts_once():
    resp = impact(
        [
            scan(serial="SAME", batch="B1", weight="3103002513"),
            scan(serial="SAME", batch="B1", weight="3103002513"),
        ],
        [rule()],
    )
    r = resp.json()["rules"][0]
    assert r["package_count"] == 1
    assert r["total_weight"] == 2.513
    assert resp.json()["unmatched"]["package_count"] == 0


def test_identical_rescan_in_digital_link_form_counts_once():
    element = scan(serial="SAME", batch="B1", weight="3103002513")
    uri = (
        "https://id.gs1.org/01/09501101530003/17/250131/21/SAME/3103/002513?10=B1"
    )
    resp = impact([element, uri], [rule()])
    assert resp.json()["rules"][0]["package_count"] == 1


def test_rescan_with_equivalent_weight_ai_counts_once():
    # 3103 002000 = 2 kg and 3104 020000 = 2 kg: same Decimal value.
    resp = impact(
        [
            scan(serial="SAME", weight="3103002000"),
            scan(serial="SAME", weight="3104020000"),
        ],
        [rule()],
    )
    assert resp.json()["rules"][0]["package_count"] == 1
    assert resp.json()["rules"][0]["total_weight"] == 2.0


def test_different_serials_are_distinct_items():
    resp = impact(
        [scan(serial="ONE", batch="B1"), scan(serial="TWO", batch="B1")],
        [rule()],
    )
    assert resp.json()["rules"][0]["package_count"] == 2


def test_same_serial_under_different_gtins_are_distinct_items():
    resp = impact(
        [scan(serial="S"), scan(gtin=GTIN2, serial="S")],
        [rule(), rule(gtin=GTIN2)],
    )
    assert [r["package_count"] for r in resp.json()["rules"]] == [1, 1]


def test_batch_conflict_returns_both_indices():
    resp = impact(
        [
            scan(serial="SAME", batch="B1"),
            scan(serial="OTHER"),
            scan(serial="SAME", batch="B2"),  # index 2 clashes with index 0
        ],
        [rule()],
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 2
    assert detail["first_scan_index"] == 0
    assert detail["error"]["error"] == "CONFLICTING_ITEM_ATTRIBUTES"
    assert "batch" in detail["error"]["message"]
    assert set(detail) == {"scan_index", "first_scan_index", "error"}


def test_expiry_conflict_returns_both_indices():
    resp = impact(
        [
            scan(serial="SAME", expiry="250131"),
            scan(serial="SAME", expiry="250228"),
        ],
        [rule()],
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 1
    assert detail["first_scan_index"] == 0
    assert "expiry" in detail["error"]["message"]


def test_different_raw_expiry_values_conflict_even_on_same_actual_date():
    # 250200 and 250228 both denote 2025-02-28, but the attributes differ.
    resp = impact(
        [
            scan(serial="SAME", expiry="250200"),
            scan(serial="SAME", expiry="250228"),
        ],
        [rule()],
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "CONFLICTING_ITEM_ATTRIBUTES"


def test_weight_conflict_returns_both_indices():
    resp = impact(
        [
            scan(serial="SAME", weight="3103002000"),  # 2 kg
            scan(serial="SAME", weight="3103003000"),  # 3 kg
        ],
        [rule()],
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 1
    assert detail["first_scan_index"] == 0
    assert "weight" in detail["error"]["message"]


def test_conflict_checks_batch_before_expiry_before_weight():
    resp = impact(
        [
            scan(serial="SAME", batch="B1", expiry="250131", weight="3103002000"),
            scan(serial="SAME", batch="B2", expiry="250228", weight="3103003000"),
        ],
        [rule()],
    )
    assert resp.json()["detail"]["error"]["message"].startswith(
        "scan 1: batch"
    )


def test_first_seen_scan_index_tracks_the_original_occurrence():
    resp = impact(
        [
            scan(serial="SAME"),
            scan(serial="SAME"),  # identical -> merged
            scan(serial="SAME", batch="X"),  # clashes with index 0, not 1
        ],
        [rule()],
    )
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 2
    assert detail["first_scan_index"] == 0


# ------------------------------------------------------- units and conversion


def test_mixed_units_converted_to_target_unit():
    # 2 kg + 25.13 lb * 0.45359237 = 13.3987762581 kg -> 13.398776
    resp = impact(
        [
            scan(serial="A", weight="3103002000"),
            scan(serial="B", weight="3202002513"),
        ],
        [rule()],
    )
    r = resp.json()["rules"][0]
    assert r["package_count"] == 2
    assert r["total_weight"] == 13.398776
    assert '"total_weight":13.398776' in resp.text


def test_lb_target_uses_exact_reciprocal():
    resp = impact(
        [scan(serial="A", weight="3100000001")],  # 1 kg
        [rule()],
        target_unit="lb",
    )
    r = resp.json()["rules"][0]
    assert r["total_weight"] == 2.204623
    assert resp.json()["target_unit"] == "lb"


def test_unmatched_weight_is_converted_and_rounded():
    resp = impact(
        [scan(serial="A", weight="3202002513")],  # 11.3987762581 kg
        [rule(gtin=GTIN2)],
    )
    assert resp.json()["unmatched"]["total_weight"] == 11.398776


def test_total_weight_renders_as_six_place_json_number():
    resp = impact([scan(serial="A", weight="3103002513")], [rule()])
    assert '"total_weight":2.513000' in resp.text
    assert '"total_weight":"2.513000"' not in resp.text


def test_half_even_rounding_only_at_response():
    # 0.0000005 kg -> nearest even at six places: 0.000000
    resp = impact([scan(serial="A", weight="3107000005")], [rule()])
    assert '"total_weight":0.000000' in resp.text
    # 0.0000015 kg -> half-even rounds up to 0.000002
    resp = impact([scan(serial="A", weight="3107000015")], [rule()])
    assert '"total_weight":0.000002' in resp.text


def test_direct_call_keeps_exact_decimal():
    from app.recall import recall_impact

    result = recall_impact(
        [scan(serial="A", weight="3202002513")],
        "kg",
        [(GTIN_OK, None, None, None)],
    )
    assert result["rules"][0]["total_weight"] == Decimal("11.3987762581")
    assert result["unmatched"]["total_weight"] == Decimal(0)
    assert not isinstance(result["rules"][0]["total_weight"], float)
    assert result["rules"][0]["package_count"] == 1


def test_direct_call_with_kg_to_lb_exact_demand_boundary():
    from app.recall import recall_impact

    result = recall_impact(
        [scan(serial="A", weight="3100000001")],
        "lb",
        [(GTIN_OK, None, None, None)],
    )
    assert result["rules"][0]["total_weight"] == KG_TO_LB


# -------------------------------------------------------------- scan errors


def test_missing_serial_rejected():
    raw = f"01{GTIN_OK}172501313103002000"
    resp = impact([raw], [rule()])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 0
    assert detail["error"]["error"] == "MISSING_AI"
    assert "21" in detail["error"]["message"]


def test_duplicate_serial_rejected():
    raw = f"01{GTIN_OK}1725013121A{GS}21B{GS}3103002000"
    resp = impact([raw], [rule()])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 0
    assert detail["error"]["error"] == "DUPLICATE_AI"


def test_missing_expiry_rejected():
    raw = f"01{GTIN_OK}21SN1{GS}3103002000"
    resp = impact([raw], [rule()])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 0
    assert detail["error"]["error"] == "MISSING_AI"
    assert "17" in detail["error"]["message"]


def test_missing_gtin_rejected():
    resp = impact([f"1725013121SN1{GS}3103002000"], [rule()])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "MISSING_AI"


def test_missing_weight_rejected():
    resp = impact([f"01{GTIN_OK}1725013121SN1"], [rule()])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"]["error"] == "MISSING_AI"
    assert "310n/320n" in detail["error"]["message"]


def test_duplicate_weight_rejected():
    resp = impact(
        [f"01{GTIN_OK}1725013121SN1{GS}31030020003202002513"],
        [rule()],
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"]["error"] == "DUPLICATE_AI"
    assert "3103" in detail["error"]["found"]
    assert "3202" in detail["error"]["found"]


def test_two_batches_rejected():
    raw = f"01{GTIN_OK}1725013110B1{GS}10B2{GS}21SN1{GS}3103002000"
    resp = impact([raw], [rule()])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "DUPLICATE_AI"


def test_parser_error_reports_scan_index_and_original_error():
    resp = impact(
        [scan(serial="A"), "01095011015300041725013121SN1\x1d3103002000"],
        [rule()],
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["scan_index"] == 1
    assert detail["error"]["error"] == "INVALID_CHECK_DIGIT"
    assert set(detail) == {"scan_index", "error"}


# -------------------------------------------------------------- rule errors


def test_empty_batches_list_is_422_with_rule_index():
    resp = impact([scan(serial="A")], [rule(), rule(batches=[])])
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["rule_index"] == 1
    assert detail["error"]["error"] == "INVALID_BATCHES"
    assert set(detail) == {"rule_index", "error"}


def test_inverted_expiry_interval_is_422_with_rule_index():
    resp = impact(
        [scan(serial="A")],
        [rule(expiry_from="2025-06-30", expiry_to="2025-01-01")],
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["rule_index"] == 0
    assert detail["error"]["error"] == "INVALID_EXPIRY_RANGE"
    assert "2025-06-30" in detail["error"]["message"]


def test_equal_interval_bounds_are_valid():
    resp = impact(
        [scan(serial="A", expiry="250131")],
        [rule(expiry_from="2025-01-31", expiry_to="2025-01-31")],
    )
    assert resp.status_code == 200
    assert resp.json()["rules"][0]["package_count"] == 1


def test_rule_validation_runs_before_scan_parsing():
    resp = impact(["not-a-gs1-string"], [rule(batches=[])])
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"]["error"] == "INVALID_BATCHES"


def test_bad_expiry_date_format_is_422():
    for bad in ("2025-13-01", "20250101", "25-01-01", "not-a-date"):
        resp = impact([], [rule(expiry_from=bad)])
        assert resp.status_code == 422, bad


def test_empty_string_inside_batches_is_422():
    resp = impact([], [rule(batches=["B1", ""])])
    assert resp.status_code == 422


def test_non_date_expiry_value_is_422():
    resp = client.post(
        "/recall-impact",
        json={"scans": [], "rules": [{"gtin": GTIN_OK, "expiry_from": 20250101}]},
    )
    assert resp.status_code == 422


# ---------------------------------------------------- request model validation


def test_missing_scans_field_is_422():
    resp = client.post("/recall-impact", json={"rules": []})
    assert resp.status_code == 422


def test_missing_rules_field_is_422():
    resp = client.post("/recall-impact", json={"scans": []})
    assert resp.status_code == 422


def test_unknown_target_unit_is_422():
    resp = impact([], [], target_unit="stone")
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "target_unit"]


def test_non_string_scan_entry_is_422():
    resp = client.post(
        "/recall-impact",
        json={"scans": [123], "rules": []},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "scans", 0]


def test_rule_missing_gtin_is_422():
    resp = client.post(
        "/recall-impact",
        json={"scans": [], "rules": [{"batches": ["B1"]}]},
    )
    assert resp.status_code == 422


def test_default_target_unit_is_kg():
    resp = impact([scan(serial="A")], [rule()])
    assert resp.json()["target_unit"] == "kg"


def test_items_are_counted_even_when_their_weight_is_zero():
    resp = impact(
        [
            scan(serial="A", weight="3103000000"),
            scan(serial="B", weight="3103000000"),
        ],
        [rule()],
    )
    r = resp.json()["rules"][0]
    assert r["package_count"] == 2
    assert r["total_weight"] == 0.0
