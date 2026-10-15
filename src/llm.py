"""Optional LLM narrative via Ollama (cloud or local). Falls back to a template when no key / no network."""
from __future__ import annotations

import json
import os

import requests

SYSTEM = ("You are a cybersecurity engineer drafting a TARA entry following ISO/SAE 21434. "
          "Use ONLY the JSON provided. Never invent numbers, standards clauses or controls. "
          "Write 150-200 words: threat summary, why the risk is rated this way, recommended actions, one caveat. "
          "End with: 'Draft for review by a qualified engineer.'")


def template_narrative(a: dict, ev: dict | None) -> str:
    det = f" The detector flagged {ev['frames']} frames between {ev['start']:.2f}s and {ev['end']:.2f}s." if ev else ""
    return (f"{a['name']} ({a['id']}, {a['stride']}) targets {a['asset'].lower()}.{det} Impact is rated {a['impact']} "
            f"and attack feasibility {a['feasibility']} (attack potential {a['potential_score']}), giving risk value "
            f"{a['risk']} ({a['risk_label']}): {a['treatment'].lower()}. Priority controls: "
            + "; ".join(m["control"] for m in a["mitigations"][:3]) + f". Safety link: {a['hazard']}. "
            "Rating tables are illustrative. Draft for review by a qualified engineer.")


def narrative(a: dict, ev: dict | None = None) -> tuple[str, str]:
    host = os.environ.get("OLLAMA_HOST", "https://ollama.com").rstrip("/")
    key = os.environ.get("OLLAMA_API_KEY", "")
    model = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
    if not key and "localhost" not in host and "127.0.0.1" not in host:
        return template_narrative(a, ev), "template (no OLLAMA_API_KEY set)"
    payload = {k: a[k] for k in ("id", "name", "stride", "asset", "damage", "impact", "feasibility", "potential_score", "risk", "risk_label", "treatment", "hazard")}
    payload["mitigations"] = [m["control"] for m in a["mitigations"]]
    payload["detection_evidence"] = ev
    try:
        r = requests.post(f"{host}/api/chat", timeout=60, headers={"Authorization": f"Bearer {key}"} if key else {},
                          json={"model": model, "stream": False, "messages": [
                              {"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload)}]})
        r.raise_for_status()
        return r.json()["message"]["content"].strip(), f"Ollama ({model})"
    except Exception as e:  # network, auth, model name ...
        return template_narrative(a, ev), f"template (LLM unavailable: {type(e).__name__})"
