#!/usr/bin/env python3
"""
Batch-test PlanAid backend or Python service directly.
Usage: python run_batch.py [--mode backend|python]
"""

import os
import json
import csv
import requests
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Config
MODE = "python"  # default
if len(sys.argv) > 1 and sys.argv[1] == "--mode=backend":
    MODE = "backend"

API_URL = {
    "backend": "http://localhost:5251/api/check-field-consistency",
    "python": "http://localhost:8000/api/check-field-consistency"
}[MODE]

# Setup folders
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
BASE_DIR = Path(__file__).parent
MOCK_DIR = BASE_DIR / "data" / "mock"
BATCH_DIR = BASE_DIR / "results" / f"batch_{timestamp}"
RESULTS_DIR = BATCH_DIR / "api_results"
DOCLING_DIR = BATCH_DIR / "docling"
METRICS_DIR = Path("metrics")

for d in [RESULTS_DIR, DOCLING_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging
LOG_FILE = BATCH_DIR / "batch_test.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info(f"🧪 Running in mode: {MODE}")


def get_test_cases() -> List[str]:
    return [d.name for d in (MOCK_DIR / "planbestemmelser").iterdir() if d.is_dir()]


def get_case_files(case: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    bestemmelser = next((p for p in (MOCK_DIR / "planbestemmelser" / case).glob("*.pdf")), None)
    plankart = next((p for p in (MOCK_DIR / "plankart" / case).glob("*.pdf")), None)
    sosi = next((p for p in (MOCK_DIR / "sosi" / case).glob("*.sos")), None)
    return bestemmelser, plankart, sosi


def call_api_direct(bestemmelser: Path, plankart: Path, sosi: Optional[Path]) -> Dict:
    with open(plankart, 'rb') as p, open(bestemmelser, 'rb') as b:
        files = {
            'plankart': (plankart.name, p.read(), 'application/pdf'),
            'bestemmelser': (bestemmelser.name, b.read(), 'application/pdf'),
        }

    if sosi:
        with open(sosi, 'rb') as s:
            files['sosi'] = (sosi.name, s.read(), 'text/plain')

    logger.info(f"→ Sending request directly to Python: {API_URL}")
    response = requests.post(API_URL, files=files)
    response.raise_for_status()
    result = response.json()
    result['processing_time'] = response.elapsed.total_seconds()
    return result


def call_api_backend(bestemmelser: Path, plankart: Path, sosi: Optional[Path]) -> Dict:
    with open(bestemmelser, 'rb') as b, open(plankart, 'rb') as p:
        files = {
            'bestemmelser': (bestemmelser.name, b, 'application/pdf'),
            'plankart': (plankart.name, p, 'application/pdf'),
        }
        if sosi:
            files['sosi'] = (sosi.name, open(sosi, 'rb'), 'text/plain')

        logger.info(f"→ Sending request to backend: {API_URL}")
        response = requests.post(API_URL, files=files)
        response.raise_for_status()
        result = response.json()
        result['processing_time'] = response.elapsed.total_seconds()

        # Close all file streams manually (important)
        for f in files.values():
            if hasattr(f[1], 'close'):
                f[1].close()

        return result


def call_api(bestemmelser: Path, plankart: Path, sosi: Optional[Path]) -> Dict:
    if MODE == "python":
        return call_api_direct(bestemmelser, plankart, sosi)
    else:
        return call_api_backend(bestemmelser, plankart, sosi)


def save_results(case: str, api_result: Dict):
    with open(RESULTS_DIR / f"{case}_result.json", "w") as f:
        json.dump(api_result, f, indent=2)

    if "docling_output" in api_result:
        with open(DOCLING_DIR / f"{case}.json", "w") as f:
            json.dump(api_result["docling_output"], f, indent=2)


def generate_summary():
    csv_path = BATCH_DIR / "summary_metrics.csv"
    rows = []

    for result_file in RESULTS_DIR.glob("*_result.json"):
        with open(result_file) as f:
            result = json.load(f)
        case = result_file.stem.replace("_result", "")
        rows.append({
            "case": case,
            "matching_fields": len(result.get("matching_fields", [])),
            "only_in_bestemmelser": len(result.get("only_in_bestemmelser", [])),
            "only_in_plankart": len(result.get("only_in_plankart", [])),
            "only_in_sosi": len(result.get("only_in_sosi", [])),
            "is_consistent": result.get("is_consistent", False),
            "processing_time": round(result.get("processing_time", 0), 2),
            "error": result.get("error", "")
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    logger.info("🔁 Starting batch testing")
    for case in get_test_cases():
        logger.info(f"📄 Case: {case}")
        bestemmelser, plankart, sosi = get_case_files(case)

        if not bestemmelser or not plankart:
            logger.warning(f"⚠️ Missing files for case: {case}")
            continue

        try:
            result = call_api(bestemmelser, plankart, sosi)
            save_results(case, result)
            logger.info(f"✅ Case done: {case}")
        except Exception as e:
            logger.error(f"❌ Failed case {case}: {e}")
            save_results(case, {"error": str(e)})

    generate_summary()
    logger.info("📊 Summary written")
    logger.info("✅ Batch complete")


if __name__ == "__main__":
    main()
