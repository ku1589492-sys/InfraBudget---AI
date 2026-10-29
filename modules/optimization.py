"""
PuLP Linear Programming & Mathematical Optimization Engine for Municipal Budget Allocation.
Solves the constrained fund distribution problem to maximize infrastructure deficit reduction
across wards while respecting equity floors, upper caps, and sectoral priority weights.
"""

import pandas as pd
import numpy as np
import time
import pulp

def run_pulp_budget_optimization(
    df: pd.DataFrame,
    total_budget_lakhs: float,
    sector_weights: dict,
    min_ward_budget: float = 40.0,
    max_ward_budget: float = 450.0
) -> dict:
    """
    Executes a PuLP Linear Programming optimization.
    Returns the optimal allocations per ward, sectoral splits, deficit improvement, and solver telemetry.
    """
    start_time = time.time()
    
    wards = df["ward_name"].tolist()
    ward_ids = df["ward_id"].tolist()
    deficit_map = dict(zip(ward_ids, df["deficit_score"]))
    pop_map = dict(zip(ward_ids, df["population"]))
    current_budget_map = dict(zip(ward_ids, df["current_budget_lakhs"]))

    # Normalize sector priority multiplier
    # e.g., if Roads has high priority, wards with higher road age get extra boost
    sector_multiplier = sum(sector_weights.values()) / max(1e-5, len(sector_weights))

    # Weight per ward based on deficit and population needs
    ward_weight_map = {}
    for wid in ward_ids:
        row = df[df["ward_id"] == wid].iloc[0]
        # Calculate composite priority coefficient
        road_priority = (row["avg_road_age_years"] / 10.0) * sector_weights.get("Roads & Pavements", 1.0)
        water_priority = (row["water_deficit_pct"] / 50.0) * sector_weights.get("Water Supply & Sewage", 1.0)
        sanitation_priority = (row["citizen_complaints_30d"] / 200.0) * sector_weights.get("Sanitation & Waste", 1.0)
        green_priority = ((45 - row["green_cover_pct"]) / 35.0) * sector_weights.get("Greenery & Lighting", 1.0)
        
        sector_impact = road_priority + water_priority + sanitation_priority + green_priority
        priority_coeff = (row["deficit_score"] / 50.0) * (row["population"] / 100000.0) * (1.0 + sector_impact * 0.25)
        ward_weight_map[wid] = max(0.5, priority_coeff)

    # Initialize PuLP Problem: Maximize Deficit Reduction
    prob = pulp.LpProblem("Municipal_Fund_Allocation", pulp.LpMaximize)

    # Decision variables: allocation to ward wid in Lakhs
    alloc_vars = {}
    for wid in ward_ids:
        alloc_vars[wid] = pulp.LpVariable(
            f"Alloc_{wid}",
            lowBound=min_ward_budget,
            upBound=max_ward_budget,
            cat=pulp.LpContinuous
        )

    # Objective: Maximize weighted marginal deficit reduction
    prob += pulp.lpSum([ward_weight_map[wid] * alloc_vars[wid] for wid in ward_ids]), "Total_Deficit_Reduction"

    # Constraint 1: Total allocated funds must not exceed total budget
    prob += pulp.lpSum([alloc_vars[wid] for wid in ward_ids]) <= total_budget_lakhs, "Total_Budget_Constraint"

    # Solve using default solver
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    
    solve_duration_ms = round((time.time() - start_time) * 1000, 2)
    solver_status_str = pulp.LpStatus[status]

    # Extract allocations
    results = []
    total_allocated = 0.0

    for wid in ward_ids:
        row = df[df["ward_id"] == wid].iloc[0]
        if solver_status_str == "Optimal":
            allocated_val = round(float(alloc_vars[wid].varValue), 2)
        else:
            # Fallback proportional logic if solver encounters edge case
            share = ward_weight_map[wid] / sum(ward_weight_map.values())
            allocated_val = round(min(max_ward_budget, max(min_ward_budget, total_budget_lakhs * share)), 2)

        total_allocated += allocated_val
        current_val = current_budget_map[wid]
        budget_delta = round(allocated_val - current_val, 2)
        
        # Calculate simulated post-allocation projected deficit score
        # Diminishing returns formula
        reduction_pct = min(55.0, (allocated_val / 250.0) * 35.0)
        projected_deficit = round(max(8.0, row["deficit_score"] * (1 - reduction_pct / 100.0)), 1)
        
        # Sectoral distribution for this ward
        sec_sum = sum(sector_weights.values())
        sec_breakdown = {
            sec: round(allocated_val * (sector_weights[sec] / sec_sum), 2)
            for sec in sector_weights
        }

        results.append({
            "ward_id": wid,
            "ward_name": row["ward_name"],
            "zone": row["zone"],
            "initial_deficit": row["deficit_score"],
            "current_budget_lakhs": current_val,
            "optimal_budget_lakhs": allocated_val,
            "budget_delta_lakhs": budget_delta,
            "projected_deficit": projected_deficit,
            "deficit_reduction_pct": round(((row["deficit_score"] - projected_deficit) / row["deficit_score"]) * 100, 1),
            **sec_breakdown
        })

    alloc_df = pd.DataFrame(results)
    
    # Summary city-wide metrics
    initial_city_avg_deficit = round(float(df["deficit_score"].mean()), 1)
    projected_city_avg_deficit = round(float(alloc_df["projected_deficit"].mean()), 1)
    city_improvement_pct = round(((initial_city_avg_deficit - projected_city_avg_deficit) / initial_city_avg_deficit) * 100, 1)

    return {
        "status": solver_status_str,
        "solve_duration_ms": solve_duration_ms,
        "total_budget_lakhs": total_budget_lakhs,
        "total_allocated_lakhs": round(total_allocated, 2),
        "unallocated_lakhs": round(max(0.0, total_budget_lakhs - total_allocated), 2),
        "initial_city_avg_deficit": initial_city_avg_deficit,
        "projected_city_avg_deficit": projected_city_avg_deficit,
        "city_improvement_pct": city_improvement_pct,
        "allocations_df": alloc_df
    }
