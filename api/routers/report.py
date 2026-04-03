"""
api/routers/report.py
=====================
Crowdsource safety report endpoint.
POST /report → stores user-reported unsafe segment
"""

import logging
import csv
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from api.models.request  import ReportRequest
from api.models.response import ReportResponse

log      = logging.getLogger("api.report")
router   = APIRouter(prefix="/report", tags=["reporting"])
REPORTS  = Path("data/raw/user_reports.csv")


@router.post("/", response_model=ReportResponse)
async def post_report(req: ReportRequest):
    """
    Records a user safety report for a road segment.
    Reports feed back into safety scores with time decay.
    """
    try:
        REPORTS.parent.mkdir(parents=True, exist_ok=True)

        write_header = not REPORTS.exists()
        with open(REPORTS, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp", "lat", "lon",
                    "category", "description", "hour"
                ])
            writer.writerow([
                datetime.now().isoformat(),
                req.lat, req.lon,
                req.category,
                req.description,
                req.hour or datetime.now().hour,
            ])

        log.info(
            f"Report recorded: ({req.lat:.4f}, {req.lon:.4f}) "
            f"- {req.category}"
        )

        return ReportResponse(
            status="recorded",
            message="Thank you. Your report helps make routes safer.",
            lat=req.lat,
            lon=req.lon,
        )

    except Exception as e:
        log.error(f"Report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/count")
async def get_report_count():
    """Returns total number of user reports recorded."""
    if not REPORTS.exists():
        return {"count": 0}
    with open(REPORTS, "r") as f:
        count = sum(1 for _ in f) - 1  # subtract header
    return {"count": max(0, count)}