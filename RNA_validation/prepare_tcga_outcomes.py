#!/usr/bin/env python3
"""Create a patient-level OS/PFS/response sidecar from raw GDC clinical JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from RNA_validation.common import clean, max_nonnegative, parse_numeric, read_tsv, require_empty_output, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tcga-count-dir", type=Path, required=True, help="standardized_tcga_raw_counts")
    p.add_argument("--clinical-root", type=Path, required=True, help="clinical_by_project")
    p.add_argument("--selected-patients", type=Path, default=None, help="Optional selected_expression_files.tsv from TCGA preparation")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit-projects", type=int, default=None, help="Deterministic smoke-test limit after sorting projects")
    p.add_argument("--limit-patients-per-project", type=int, default=None, help="Deterministic smoke-test patient limit")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def response_class(value: object) -> str | None:
    text = clean(value).upper()
    mapping = {
        "COMPLETE RESPONSE": "CR", "CR-COMPLETE RESPONSE": "CR", "CR": "CR",
        "PARTIAL RESPONSE": "PR", "PR-PARTIAL RESPONSE": "PR", "PR": "PR",
        "STABLE DISEASE": "SD", "SD-STABLE DISEASE": "SD", "SD": "SD",
        "PROGRESSIVE DISEASE": "PD", "PD-PROGRESSIVE DISEASE": "PD", "PD": "PD",
    }
    return mapping.get(text)


def truthy_progression(value: object) -> bool:
    return clean(value).lower() in {"yes", "true", "1", "progression", "recurrence", "progression or recurrence"}


def flatten_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    data = payload.get("data", payload)
    if isinstance(data, dict) and "hits" in data:
        return data["hits"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected clinical JSON structure: {path}")


def derive(case: dict, patient_id: str, dataset_id: str, project_id: str) -> dict:
    demographic = case.get("demographic") or {}
    diagnoses = case.get("diagnoses") or []
    followups = case.get("follow_ups") or []
    vital = clean(demographic.get("vital_status"))
    death = parse_numeric(demographic.get("days_to_death"))
    followup_times = []
    progression_times = []
    progression_without_time = False
    recurrence_times = []
    responses = set()
    agents = set()
    treatment_outcomes = []
    prior_treatment = set()
    stages = []
    for diagnosis in diagnoses:
        followup_times.extend([diagnosis.get("days_to_last_follow_up"), diagnosis.get("days_to_last_known_disease_status")])
        recurrence_times.append(diagnosis.get("days_to_recurrence"))
        if truthy_progression(diagnosis.get("progression_or_recurrence")) and parse_numeric(diagnosis.get("days_to_recurrence")) is None:
            progression_without_time = True
        if clean(diagnosis.get("prior_treatment")):
            prior_treatment.add(clean(diagnosis.get("prior_treatment")))
        if clean(diagnosis.get("ajcc_pathologic_stage")):
            stages.append(clean(diagnosis.get("ajcc_pathologic_stage")))
        for treatment in diagnosis.get("treatments") or []:
            outcome = clean(treatment.get("treatment_outcome"))
            if outcome:
                treatment_outcomes.append(outcome)
                category = response_class(outcome)
                if category:
                    responses.add(category)
            if clean(treatment.get("therapeutic_agents")):
                agents.add(clean(treatment.get("therapeutic_agents")))
    for followup in followups:
        followup_times.append(followup.get("days_to_follow_up"))
        progression_times.extend([followup.get("days_to_progression"), followup.get("days_to_progression_free")])
        recurrence_times.append(followup.get("days_to_recurrence"))
        if truthy_progression(followup.get("progression_or_recurrence")) and all(
            parse_numeric(followup.get(k)) is None for k in ("days_to_progression", "days_to_progression_free", "days_to_recurrence")
        ):
            progression_without_time = True
        category = response_class(followup.get("disease_response"))
        if category:
            responses.add(category)
    censor = max_nonnegative(followup_times)
    if vital.lower() == "dead" and death is not None:
        os_time, os_event, os_rule = death, 1, "dead_days_to_death"
    elif vital.lower() == "alive" and censor is not None:
        os_time, os_event, os_rule = censor, 0, "alive_max_followup"
    else:
        os_time, os_event, os_rule = np.nan, np.nan, "insufficient"
    event_times = [x for x in [parse_numeric(v) for v in progression_times + recurrence_times] if x is not None]
    if death is not None:
        event_times.append(death)
    if event_times:
        pfs_time, pfs_event, pfs_rule = min(event_times), 1, "earliest_progression_recurrence_or_death"
    elif censor is not None and not progression_without_time:
        pfs_time, pfs_event, pfs_rule = censor, 0, "censored_at_max_followup_no_timed_event"
    else:
        pfs_time, pfs_event, pfs_rule = np.nan, np.nan, "insufficient_or_untimed_event"
    if len(responses) == 1:
        response = next(iter(responses))
        response_group = "Response" if response in {"CR", "PR"} else "Non-response"
        response_review = "no"
    elif len(responses) > 1:
        response, response_group, response_review = "|".join(sorted(responses)), "NA", "yes_conflicting"
    else:
        response, response_group, response_review = "NA", "NA", "yes_missing"
    return {
        "patient_id": patient_id,
        "source_patient_id": clean(case.get("submitter_id")),
        "dataset_id": dataset_id,
        "source_project_id": project_id,
        "age_at_index": clean(demographic.get("age_at_index")) or "NA",
        "sex": clean(demographic.get("sex_at_birth")) or "NA",
        "pathologic_stage": stages[0] if stages else "NA",
        "vital_status": vital or "NA",
        "os_time_days": os_time,
        "os_event": os_event,
        "os_derivation_rule": os_rule,
        "pfs_time_days_candidate": pfs_time,
        "pfs_event_candidate": pfs_event,
        "pfs_derivation_rule": pfs_rule,
        "pfs_needs_review": "yes" if progression_without_time or pfs_rule.startswith("censored") else "no",
        "response_category": response,
        "response_group": response_group,
        "response_needs_review": response_review,
        "treatment_outcomes_source": "|".join(sorted(set(treatment_outcomes))) or "NA",
        "therapeutic_agents_source": "|".join(sorted(agents)) or "NA",
        "prior_treatment_source": "|".join(sorted(prior_treatment)) or "NA",
    }


def main() -> None:
    args = parse_args()
    for name in ("limit_projects", "limit_patients_per_project"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    out = require_empty_output(args.output_dir, args.overwrite)
    patients = read_tsv(args.tcga_count_dir / "patients.tsv")
    if args.selected_patients is not None:
        selected = read_tsv(args.selected_patients)
        selected_ids = set(selected["patient_id"])
        patients = patients.loc[patients["patient_id"].isin(selected_ids)].copy()
        missing = selected_ids - set(patients["patient_id"])
        if missing:
            raise ValueError(f"Selected sidecar contains {len(missing)} patients absent from TCGA patients.tsv")
    else:
        if args.limit_projects is not None:
            projects = sorted(patients["source_project_id"].unique())[:args.limit_projects]
            patients = patients.loc[patients["source_project_id"].isin(projects)].copy()
        if args.limit_patients_per_project is not None:
            patients = patients.sort_values(["source_project_id", "patient_id"], kind="stable")
            patients = patients.groupby("source_project_id", sort=True, group_keys=False).head(args.limit_patients_per_project).copy()
    rows = []
    for project_id, project_patients in patients.groupby("source_project_id", sort=True):
        path = args.clinical_root / project_id / f"{project_id}_clinical_raw.json"
        cases = {clean(c.get("submitter_id")): c for c in flatten_cases(path)}
        for patient in project_patients.itertuples(index=False):
            case = cases.get(patient.source_patient_id)
            if case is None:
                raise ValueError(f"Clinical case missing for {patient.source_patient_id}")
            rows.append(derive(case, patient.patient_id, patient.dataset_id, project_id))
    table = pd.DataFrame(rows)
    table.to_csv(out / "tcga_patient_outcomes.tsv", sep="\t", index=False)
    write_json(out / "summary.json", {
        "n_patients": len(table),
        "os_evaluable": int(table["os_time_days"].notna().sum()),
        "os_events": int(pd.to_numeric(table["os_event"], errors="coerce").eq(1).sum()),
        "pfs_candidate_evaluable": int(table["pfs_time_days_candidate"].notna().sum()),
        "response_unambiguous": int(table["response_group"].isin(["Response", "Non-response"]).sum()),
        "selected_patients": str(args.selected_patients) if args.selected_patients else None,
        "limit_projects": args.limit_projects,
        "limit_patients_per_project": args.limit_patients_per_project,
        "warning": "PFS and response are derived candidates from heterogeneous raw GDC records and retain review flags.",
    })
    print(f"outcomes={len(table)} output={out}")


if __name__ == "__main__":
    main()
