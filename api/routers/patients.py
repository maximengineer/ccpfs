"""GET /api/patients/{index}/curve - serve individual patient data."""

from fastapi import APIRouter, HTTPException

from api.dependencies import store
from api.schemas import PatientCurve
from config import C_EVENT, C_VISIT, SPECIALTY_NAMES

router = APIRouter()


@router.get("/patients/{index}/curve", response_model=PatientCurve)
def get_patient_curve(index: int):
    n = store.curves.shape[0]
    if index < 0 or index >= n:
        raise HTTPException(status_code=404, detail=f"Patient index must be 0-{n-1}")

    curve = store.curves[index].tolist()
    event = bool(store.e_test[index])
    time = float(store.t_test[index])

    # Get assigned day from best available policy
    assigned_day = 15  # fallback
    if store.scheduling and "mincost_specialty" in store.scheduling:
        assigned_day = int(store.scheduling["mincost_specialty"][index])
    elif store.scheduling and "greedy_specialty" in store.scheduling:
        assigned_day = int(store.scheduling["greedy_specialty"][index])

    # Get specialty from cohort (same positional index as curves)
    specialty_idx = int(store.cohort_test["specialty_pool"][index])
    specialty_name = SPECIALTY_NAMES[specialty_idx] if specialty_idx < len(SPECIALTY_NAMES) else "unknown"

    # Cost at assigned day
    cost = C_EVENT * (1.0 - curve[assigned_day]) + C_VISIT

    return PatientCurve(
        patient_index=index,
        survival_curve=curve,
        assigned_day=assigned_day,
        specialty=specialty_name,
        specialty_index=specialty_idx,
        event_indicator=event,
        time_to_event=round(time, 2),
        cost=round(cost, 0),
    )


@router.get("/patients/count")
def get_patient_count():
    return {"count": store.curves.shape[0]}
