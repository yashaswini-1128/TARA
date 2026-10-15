"""Rule-based TARA engine in the spirit of ISO/SAE 21434 (clause 15).

Deliberately DETERMINISTIC: impact x attack-feasibility -> risk value comes from fixed
tables, not from a language model (same principle as the case-study document).
Tables hold ILLUSTRATIVE values - calibrate them against your organisation's
authorised copy of the standard and risk policy before any real use.
"""
from __future__ import annotations

IMPACT_LEVELS = ["Negligible", "Moderate", "Major", "Severe"]
FEAS_LEVELS = ["Very low", "Low", "Medium", "High"]
# rows: impact, columns: attack feasibility (Very low .. High) -> risk value 1..5
RISK_MATRIX = {
    "Negligible": [1, 1, 1, 1],
    "Moderate":   [1, 2, 2, 3],
    "Major":      [1, 2, 3, 4],
    "Severe":     [2, 3, 4, 5],
}
# attack-potential factors (ISO 18045 style): factor -> {level: points}
POTENTIAL = {
    "Elapsed time": {"<= 1 day": 0, "<= 1 week": 1, "<= 1 month": 4, "<= 6 months": 17, "> 6 months": 19},
    "Expertise": {"Layman": 0, "Proficient": 3, "Expert": 6, "Multiple experts": 8},
    "Knowledge of item": {"Public": 0, "Restricted": 3, "Confidential": 7, "Strictly confidential": 11},
    "Window of opportunity": {"Unlimited": 0, "Easy": 1, "Moderate": 4, "Difficult": 10},
    "Equipment": {"Standard": 0, "Specialized": 4, "Bespoke": 7, "Multiple bespoke": 9},
}
RISK_LABEL = {1: "Very low", 2: "Low", 3: "Medium", 4: "High", 5: "Critical"}
RISK_COLOR = {1: "#34D399", 2: "#A3E635", 3: "#FBBF24", 4: "#FB923C", 5: "#F87171"}


def feasibility_from_score(score: int) -> str:
    """Higher attack potential needed -> lower feasibility."""
    if score <= 13:
        return "High"
    if score <= 19:
        return "Medium"
    if score <= 24:
        return "Low"
    return "Very low"


def risk_value(impact: str, feasibility: str) -> int:
    return RISK_MATRIX[impact][FEAS_LEVELS.index(feasibility)]


def treatment(risk: int) -> str:
    if risk >= 4:
        return "Reduce - mitigation mandatory before release"
    if risk == 3:
        return "Reduce - mitigation recommended"
    if risk == 2:
        return "Retain with monitoring"
    return "Retain"


def _m(mid, control, layer, uplift, drop, detail):
    return {"id": mid, "control": control, "layer": layer, "uplift": uplift, "impact_drop": drop, "detail": detail}


THREATS = {
    "DoS": {
        "id": "TS-01", "name": "CAN bus flooding with highest-priority frames", "stride": "Denial of Service",
        "asset": "Powertrain CAN availability and the ECUs that depend on timely messages",
        "interface": "OBD-II port; compromised telematics / infotainment gateway",
        "damage": "Starved legitimate messages -> loss of powertrain / ADAS functions, forced limp-home mode.",
        "impact": {"Safety": "Major", "Financial": "Moderate", "Operational": "Major", "Privacy": "Negligible"},
        "potential": {"Elapsed time": "<= 1 day", "Expertise": "Layman", "Knowledge of item": "Public",
                      "Window of opportunity": "Easy", "Equipment": "Standard"},
        "hazard": "Loss of drive torque / unintended deceleration (candidate hazard for the HARA team)",
        "goal": "CG-01: Powertrain CAN communication shall remain available under frame-flooding attacks.",
        "mitigations": [
            _m("M1", "Gateway rate limiting + ID filtering", "Preventive", 6, 0, "Drop frames above per-ID rate; block ID 0x000 from external segments."),
            _m("M2", "Authenticated diagnostic gateway for OBD-II", "Preventive", 8, 0, "Raises attacker effort for bus access (UDS security access, session control)."),
            _m("M3", "Network IDS with alerting (this model)", "Detective", 0, 0, "Detects flooding within milliseconds and raises a DTC."),
            _m("M4", "Bus-off recovery + degraded-mode strategy", "Responsive", 0, 1, "Keeps a safe minimal function set alive, reducing effect severity."),
        ],
        "tests": ["HIL: flood at 90% bus load; verify rate limiter and degraded mode engage within the FTTI.",
                  "Fuzz-free regression: confirm legitimate 10 ms messages keep their deadlines under flood."],
    },
    "Fuzzy": {
        "id": "TS-02", "name": "Random frame injection (fuzzing)", "stride": "Tampering / Denial of Service",
        "asset": "ECU message handling and actuator control logic",
        "interface": "OBD-II port; compromised ECU on a connected segment",
        "damage": "Unintended actuation, ECU faults and DTC storms, unpredictable vehicle behaviour.",
        "impact": {"Safety": "Major", "Financial": "Moderate", "Operational": "Moderate", "Privacy": "Negligible"},
        "potential": {"Elapsed time": "<= 1 week", "Expertise": "Proficient", "Knowledge of item": "Public",
                      "Window of opportunity": "Easy", "Equipment": "Standard"},
        "hazard": "Unintended actuation of vehicle functions (candidate hazard for the HARA team)",
        "goal": "CG-02: ECUs shall accept only authenticated, plausible messages from authorised senders.",
        "mitigations": [
            _m("M1", "Gateway ID whitelist", "Preventive", 4, 0, "Unknown IDs never cross into the powertrain segment."),
            _m("M2", "SecOC message authentication (MAC + freshness)", "Preventive", 10, 0, "Random payloads fail MAC verification."),
            _m("M3", "Network IDS with alerting (this model)", "Detective", 0, 0, "Flags unknown IDs and entropy anomalies."),
            _m("M4", "Receiver-side plausibility checks", "Responsive", 0, 1, "Ignore out-of-range values, limiting actuation impact."),
        ],
        "tests": ["Protocol fuzzing campaign on the bench; no unintended actuation, no ECU reset.",
                  "Verify unknown-ID frames are dropped at the gateway."],
    },
    "Spoof_RPM": {
        "id": "TS-03", "name": "Forged engine-speed (RPM) messages", "stride": "Spoofing / Tampering",
        "asset": "Integrity of the engine-speed signal consumed by cluster, TCM and stability control",
        "interface": "OBD-II port; compromised ECU impersonating the engine controller",
        "damage": "False RPM -> wrong shift decisions, misleading cluster, mis-estimated engine state.",
        "impact": {"Safety": "Major", "Financial": "Moderate", "Operational": "Moderate", "Privacy": "Negligible"},
        "potential": {"Elapsed time": "<= 1 month", "Expertise": "Proficient", "Knowledge of item": "Restricted",
                      "Window of opportunity": "Moderate", "Equipment": "Standard"},
        "hazard": "Unintended gear shift / torque request from false engine speed (candidate hazard for the HARA team)",
        "goal": "CG-03: Safety-relevant signals shall be authenticated and checked for freshness and plausibility.",
        "mitigations": [
            _m("M1", "SecOC on the RPM message (MAC + freshness value)", "Preventive", 12, 0, "Forged frames without the key are rejected."),
            _m("M2", "Gateway source-ECU binding", "Preventive", 4, 0, "ID 0x316 only accepted from the engine-controller port."),
            _m("M3", "Network IDS with alerting (this model)", "Detective", 0, 0, "Detects duplicate-ID timing and frozen payloads."),
            _m("M4", "Signal plausibility monitoring (rate of change)", "Responsive", 0, 1, "Receivers fall back to a safe default on implausible RPM."),
        ],
        "tests": ["Replay forged 0x316 frames at 3x rate; verify MAC rejection and IDS alert.",
                  "Inject implausible RPM jump; verify receiver fallback."],
    },
    "Spoof_Gear": {
        "id": "TS-04", "name": "Forged gear-position messages", "stride": "Spoofing / Tampering",
        "asset": "Integrity of the gear / drive-state signal used by safety interlocks",
        "interface": "OBD-II port; compromised ECU impersonating the transmission controller",
        "damage": "ECUs believe the vehicle is in another drive state -> interlocks bypassed, unintended motion.",
        "impact": {"Safety": "Severe", "Financial": "Moderate", "Operational": "Major", "Privacy": "Negligible"},
        "potential": {"Elapsed time": "<= 1 month", "Expertise": "Proficient", "Knowledge of item": "Restricted",
                      "Window of opportunity": "Moderate", "Equipment": "Standard"},
        "hazard": "Unintended vehicle motion due to wrong drive state (candidate hazard for the HARA team)",
        "goal": "CG-04: Drive-state signals shall be authenticated; interlocks shall not rely on a single unauthenticated source.",
        "mitigations": [
            _m("M1", "SecOC on the gear-position message", "Preventive", 12, 0, "Forged drive-state frames are rejected."),
            _m("M2", "Gateway source-ECU binding", "Preventive", 4, 0, "ID 0x43F only accepted from the transmission port."),
            _m("M3", "Network IDS with alerting (this model)", "Detective", 0, 0, "Detects duplicate-ID timing and frozen payloads."),
            _m("M4", "Redundant drive-state cross-check (wheel speed vs gear)", "Responsive", 0, 1, "Interlocks use two independent sources."),
        ],
        "tests": ["Inject forged 'Park' frames while moving; verify interlocks hold and alert is raised.",
                  "Verify MAC rejection of forged 0x43F frames on the bench."],
    },
}


def _worst(impact: dict) -> str:
    return max(impact.values(), key=IMPACT_LEVELS.index)


def assess(attack: str, selected_mitigations: list[str] | None = None) -> dict | None:
    """Full assessment for one attack class; optionally apply mitigations to get the residual risk."""
    t = THREATS.get(attack)
    if t is None:
        return None
    score = sum(POTENTIAL[f][lvl] for f, lvl in t["potential"].items())
    impact = _worst(t["impact"])
    feas = feasibility_from_score(score)
    risk = risk_value(impact, feas)
    chosen = [m for m in t["mitigations"] if selected_mitigations is None or m["id"] in selected_mitigations]
    res_score = score + sum(m["uplift"] for m in chosen)
    res_impact = IMPACT_LEVELS[max(IMPACT_LEVELS.index(impact) - sum(m["impact_drop"] for m in chosen), 0)]
    res_feas = feasibility_from_score(res_score)
    return {
        "attack": attack, **{k: t[k] for k in ("id", "name", "stride", "asset", "interface", "damage", "hazard", "goal", "tests", "mitigations")},
        "impact_by_category": t["impact"], "impact": impact, "potential": t["potential"], "potential_score": score,
        "feasibility": feas, "risk": risk, "risk_label": RISK_LABEL[risk], "treatment": treatment(risk),
        "residual": {"potential_score": res_score, "impact": res_impact, "feasibility": res_feas,
                     "risk": risk_value(res_impact, res_feas), "applied": [m["id"] for m in chosen]},
    }


def all_assessments() -> list[dict]:
    return [assess(a, []) for a in THREATS]


def to_markdown(a: dict, evidence: dict | None = None) -> str:
    L = [f"# TARA entry {a['id']} - {a['name']}", "",
         f"- **STRIDE:** {a['stride']}", f"- **Asset:** {a['asset']}", f"- **Attack interface:** {a['interface']}",
         f"- **Damage scenario:** {a['damage']}",
         f"- **Impact:** {a['impact']} (" + ", ".join(f"{k}: {v}" for k, v in a['impact_by_category'].items()) + ")",
         f"- **Attack potential:** {a['potential_score']} -> feasibility **{a['feasibility']}**",
         f"- **Risk value:** {a['risk']} ({a['risk_label']}) - {a['treatment']}",
         f"- **Cybersecurity goal:** {a['goal']}", f"- **Linked safety hazard:** {a['hazard']}", ""]
    if evidence:
        L += ["## Detection evidence", *(f"- {k}: {v}" for k, v in evidence.items()), ""]
    L += ["## Mitigations", *(f"- [{m['layer']}] {m['control']}: {m['detail']}" for m in a["mitigations"]), "",
          "## Verification", *(f"- {t}" for t in a["tests"]), "",
          "_AI-assisted draft. Illustrative rating tables; requires review and approval by a qualified cybersecurity engineer._"]
    return "\n".join(L)
