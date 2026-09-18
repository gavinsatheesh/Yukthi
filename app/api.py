"""
app/api.py
FastAPI backend service exposing Chiller Forensics insights and investigation cases.

RUN COMMAND:
    uvicorn app.api:app --reload --port 8000
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

app = FastAPI(
    title="Chiller Forensics API",
    description="Intelligent Energy & Equipment Monitoring contextual anomaly investigation engine.",
    version="1.0.0",
)

# Enable CORS for frontend hackathon dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_json_file(filename: str) -> Any:
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Output artifact {filename} not found. Please run 'python ml/pipeline.py' first.",
        )
    with open(path, "r") as f:
        return json.load(f)


@app.get("/")
def root():
    return {
        "service": "Chiller Forensics API",
        "status": "online",
        "endpoints": [
            "/chillers",
            "/chillers/{equipment_id}",
            "/chillers/{equipment_id}/timeline",
            "/chillers/{equipment_id}/investigations",
            "/investigations",
            "/investigations/{case_id}",
        ],
    }


@app.get("/chillers")
def get_all_chillers():
    """Retrieve operational health summaries for all chillers."""
    return load_json_file("chiller_summary.json")


@app.get("/chillers/{equipment_id}")
def get_chiller_detail(equipment_id: str):
    """Retrieve summary for a specific chiller."""
    summaries = load_json_file("chiller_summary.json")
    for sm in summaries:
        if sm["equipment"].upper() == equipment_id.upper():
            return sm
    raise HTTPException(status_code=404, detail=f"Chiller {equipment_id} not found")


@app.get("/chillers/{equipment_id}/timeline")
def get_chiller_timeline(
    equipment_id: str,
    limit: Optional[int] = Query(default=1000, description="Max observations to return"),
):
    """
    Retrieve chronological anomaly timeline replay for frontend visualization.
    Shows NORMAL -> WATCH -> INVESTIGATE -> PRIORITY progression.
    """
    timelines = load_json_file("timeline_replay.json")
    for eq, records in timelines.items():
        if eq.upper() == equipment_id.upper():
            return {
                "equipment": eq,
                "total_records": len(records),
                "timeline": records[-limit:],
            }
    raise HTTPException(status_code=404, detail=f"Chiller {equipment_id} timeline not found")


@app.get("/investigations")
def get_all_investigations(
    severity: Optional[str] = None,
    equipment: Optional[str] = None,
):
    """Retrieve structured investigation cases with contextual evidence."""
    data = load_json_file("anomaly_results.json")
    cases = data.get("investigation_cases", [])

    if severity and isinstance(severity, str):
        cases = [c for c in cases if c["severity"].upper() == severity.upper()]
    if equipment and isinstance(equipment, str):
        cases = [c for c in cases if c["equipment"].upper() == equipment.upper()]

    return {
        "total_cases": len(cases),
        "cases": cases,
    }


@app.get("/investigations/{case_id}")
def get_investigation_case(case_id: str):
    """Retrieve details and context evidence for a specific investigation case."""
    data = load_json_file("anomaly_results.json")
    cases = data.get("investigation_cases", [])
    for c in cases:
        if c["case_id"].upper() == case_id.upper():
            return c
    raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
