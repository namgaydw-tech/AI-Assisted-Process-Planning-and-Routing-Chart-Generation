"""Transparent machining routing research baseline.

NOT a CAM engine: times are deterministic *illustrative estimates*, not machine-verified.
Optimization minimizes sequential processing + tool/machine switch penalties under
per-feature precedence and machine availability constraints.
"""
from __future__ import annotations
from functools import lru_cache
from math import pi
from app.models import Feature, Operation, PlanRequest, PlanResponse, Route, RouteSummary

TOOL_CHANGE_MIN = 1.5
MACHINE_CHANGE_MIN = 3.0


def _choose_machine(allowed: tuple[str, ...], available: list[str]) -> str:
    for option in allowed:
        if option in available:
            return option
    raise ValueError(f"No eligible available machine among {allowed}")


def _time(action: str, feat: Feature, material: str) -> float:
    """Planning heuristic only; requires calibration against machining trials."""
    hardness = 1.7 if material == "steel" else 1.0
    if action == "face_milling":
        base = 1.8 + feat.quantity * feat.length_mm * feat.width_mm / 1300
    elif action == "spot_drilling":
        base = 0.45 * feat.quantity
    elif action == "drilling":
        base = feat.quantity * (0.8 + feat.depth_mm / 32 + feat.diameter_mm / 24)
    elif action == "rough_pocket":
        base = 1.5 + feat.quantity * feat.length_mm * feat.width_mm * feat.depth_mm / 20000
    elif action == "finish_pocket":
        base = 0.7 + feat.quantity * (feat.length_mm + feat.width_mm) / 75
    elif action == "slot_milling":
        base = 0.9 + feat.quantity * feat.length_mm * feat.depth_mm / 650
    else:
        raise ValueError(f"Unknown process: {action}")
    return round(base * hardness, 2)


def build_operations(request: PlanRequest) -> list[Operation]:
    out: list[Operation] = []
    available = request.available_machines
    tools = request.available_tools
    for f in request.features:
        sequence: list[tuple[str, str, tuple[str, ...], str]]
        if f.kind == "planar_face":
            sequence = [("face_milling", "face_mill", ("cnc_mill",), "Planar face milling")]
        elif f.kind == "hole":
            sequence = [
                ("spot_drilling", "spot_drill", ("drill_press", "cnc_mill"), "Spot drill hole location"),
                ("drilling", "twist_drill", ("drill_press", "cnc_mill"), "Drill to specified nominal depth"),
            ]
        elif f.kind == "pocket":
            sequence = [
                ("rough_pocket", "end_mill", ("cnc_mill",), "Rough mill supported pocket"),
                ("finish_pocket", "end_mill", ("cnc_mill",), "Finish pocket perimeter"),
            ]
        elif f.kind == "slot":
            sequence = [("slot_milling", "slot_mill", ("cnc_mill",), "Mill a supported straight slot")]
        else:
            raise ValueError(f"Unsupported manufacturing feature: {f.kind}")
        prev: str | None = None
        for idx, (action, tool, allowed, note) in enumerate(sequence, start=1):
            if tool not in tools:
                raise ValueError(f"Missing tool '{tool}' required by feature {f.id}")
            machine = _choose_machine(allowed, available)
            opid = f"{f.id}-{idx}"
            out.append(Operation(
                id=opid, feature_id=f.id, action=action, machine=machine, tool=tool,
                process_minutes=_time(action, f, request.material),
                predecessors=[prev] if prev else [], note=note,
            ))
            prev = opid
    return out


def _trans(previous: Operation | None, current: Operation) -> tuple[float, int, int]:
    if previous is None:
        return 0.0, 0, 0
    tc = int(previous.tool != current.tool)
    mc = int(previous.machine != current.machine)
    return tc * TOOL_CHANGE_MIN + mc * MACHINE_CHANGE_MIN, tc, mc


def _route(ops: list[Operation], method: str) -> Route:
    proc = sum(x.process_minutes for x in ops)
    change, tools, machines = 0.0, 0, 0
    for a, b in zip(ops, ops[1:]):
        delta, tc, mc = _trans(a, b)
        change += delta
        tools += tc
        machines += mc
    return Route(operations=ops, summary=RouteSummary(
        process_minutes=round(proc, 2), changeover_minutes=round(change, 2),
        total_estimated_minutes=round(proc + change, 2), tool_changes=tools,
        machine_changes=machines, operation_count=len(ops), method=method,
    ))


def _greedy(ops: list[Operation]) -> list[Operation]:
    """Deterministic ready-task heuristic for large instances."""
    by_id = {o.id: o for o in ops}
    done: set[str] = set()
    ordered: list[Operation] = []
    while len(done) < len(ops):
        ready = [o for o in ops if o.id not in done and all(p in done for p in o.predecessors)]
        if not ready:
            raise ValueError("Cyclic or invalid operation precedence")
        prev = ordered[-1] if ordered else None
        selected = min(ready, key=lambda o: (_trans(prev, o)[0], o.id))
        ordered.append(by_id[selected.id])
        done.add(selected.id)
    return ordered


def _optimal_small(ops: list[Operation]) -> list[Operation]:
    """Exact dynamic programming on <= 14 operations; no ML involved."""
    n = len(ops)
    index = {o.id: i for i, o in enumerate(ops)}
    prereq = []
    for op in ops:
        bits = 0
        for pred in op.predecessors:
            if pred not in index:
                raise ValueError(f"Missing predecessor '{pred}'")
            bits |= 1 << index[pred]
        prereq.append(bits)
    full = (1 << n) - 1

    @lru_cache(None)
    def solve(done: int, last: int) -> tuple[float, tuple[int, ...]]:
        if done == full:
            return 0.0, ()
        candidates: list[tuple[float, tuple[int, ...]]] = []
        for i in range(n):
            if done & (1 << i) or (prereq[i] & done) != prereq[i]:
                continue
            delta = _trans(ops[last] if last >= 0 else None, ops[i])[0]
            rest, path = solve(done | (1 << i), i)
            candidates.append((round(delta + rest, 5), (i,) + path))
        if not candidates:
            raise ValueError("Cyclic or invalid operation precedence")
        return min(candidates)

    _cost, path = solve(0, -1)
    return [ops[i] for i in path]


def plan(request: PlanRequest) -> PlanResponse:
    ops = build_operations(request)
    baseline = _route(ops, "feature-order baseline")
    if len(ops) <= 14:
        proposed = _optimal_small(ops)
        method = "exact precedence-constrained DP (sequential operations)"
    else:
        proposed = _greedy(ops)
        method = "greedy ready-operation heuristic (>14 operations)"
    optimized = _route(proposed, method)
    # Never report an optimizer worse than the baseline.
    if optimized.summary.total_estimated_minutes > baseline.summary.total_estimated_minutes:
        optimized = baseline.model_copy(deep=True)
        optimized.summary.method = "baseline retained because heuristic was worse"
    base = baseline.summary.total_estimated_minutes
    improvement = 100 * (base - optimized.summary.total_estimated_minutes) / base if base else 0.0
    return PlanResponse(
        part_name=request.part_name, material=request.material,
        baseline=baseline, optimized=optimized,
        improvement_percent=round(improvement, 2),
        scope_note=("Research baseline only. Features are manually described, machining times "
                    "are uncalibrated heuristic estimates, and operations are scheduled "
                    "sequentially. No automatic feature recognition or machine-specific "
                    "cutting validation has been performed."),
    )
