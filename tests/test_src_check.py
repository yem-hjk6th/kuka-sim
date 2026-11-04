"""Static checker of KRL programs (kukasim/src_check.py, scripts/check_src.py).

Synthetic programs are written here; they hold no site data. Positions are relative to
the approach pose of the committed Hilbert corner program (a point over the bed).
"""
import json
import re

import numpy as np
import pytest
import yaml

from kukasim.config import REPO
from kukasim.src_check import SrcChecker, check_file, check_text, output_dir, parse_program

RESULTS = REPO / "results"
SRC_FILES = sorted(RESULTS.glob("*/*.src"))

# Tolerance against export_report.json: the report rounds clearances to 0.01 mm and speeds to
# 0.1 mm/s, the .src rounds targets to 0.001 mm / 0.0001 deg (joint changes ~1e-6 rad, far below
# 0.01 mm at the tool), and the checker samples PTP / held-orientation LINs exactly as the exporter
# (same TransitPlanner.ptp_samples and lin_sample_mm). 0.05 mm / 0.5 mm/s covers the rounding
# with margin; any modelling difference would be larger.
TOL_CLEARANCE_MM = 0.05
TOL_SPEED_MM_S = 0.5

Q0 = (55.6749, -36.0731, 108.0322, 57.0073, -79.9426, -15.0555)   # results/hilbert_corner approach PTP


@pytest.fixture(scope="module")
def checker():
    return SrcChecker()


@pytest.fixture(scope="module")
def p0(checker):
    x, y, z = checker.robot.fk_xyzabc(np.array(Q0))[:3]
    return x, y, z, checker.cfg.bed()["top_z"] + 1.3


def lin(x, y, z, a=0, b=90, c=0, extra=""):
    return f"LIN {{X {x:.3f}, Y {y:.3f}, Z {z:.3f}, A {a:g}, B {b:g}, C {c:g}}}{extra}"


def program(body, vel_cp=0.012, tool=3, head=()):
    return "\n".join(
        ["DEF synthetic()", "BAS(#INITMOV,0)", f"BAS(#TOOL,{tool})", "BAS(#BASE,0)", f"$VEL.CP = {vel_cp}",
         "$APO.CDIS = 0.5", "BAS(#VEL_PTP,5)"] + [f"$ACC_AXIS[{i}] = 10" for i in range(1, 7)] + list(head)
        + ["PTP {A1 %g, A2 %g, A3 %g, A4 %g, A5 %g, A6 %g}" % Q0] + list(body) + ["END"]) + "\n"


def clean_body(p0):
    x, y, z, zp = p0
    return [lin(x, y, zp), lin(x + 40, y, zp, extra=" C_DIS"), lin(x + 40, y + 40, zp, extra=" C_DIS"),
            lin(x, y + 40, zp, extra=" C_DIS"), lin(x, y, zp), lin(x, y, z),
            "BAS(#VEL_PTP,2)",   # an A1 swing at 5 % peaks near 200 mm/s in the model
            "PTP {A1 %g, A2 %g, A3 %g, A4 %g, A5 %g, A6 %g}" % (Q0[0] + 5, *Q0[1:])]


def codes(rep):
    return {f["code"] for f in rep["flags"]} | {i["code"] for i in rep["issues"]}


# ---------------------------------------------------------------- a. committed programs
@pytest.fixture(scope="module")
def results_reports(checker):
    return {f: check_text(f.read_text(), f.name, checker) for f in SRC_FILES}


def test_results_programs_parse_without_uninterpreted_motion(results_reports):
    assert len(SRC_FILES) >= 6
    for f, rep in results_reports.items():
        bad = [s for s in rep["statements"] if s["kind"] == "not_interpreted"]
        assert not bad, f"{f.name}: not interpreted {bad[:3]}"
        assert not any(i["severity"] == "error" for i in rep["issues"]), f.name
        n_motion = sum(1 for ln in f.read_text().splitlines() if re.match(r"\s*(PTP|LIN)\b", ln))
        assert len(rep["motions"]) == n_motion, f.name


def _transit_moves(src, rep):
    """(transit index, move label, motion record) from the exporter's '; <KIND> <label>' comments."""
    by_line = {r["line"]: r for r in rep["motions"]}
    out, tr, label = [], None, None
    for i, ln in enumerate(src.read_text().splitlines(), start=1):
        m = re.match(r"\s*; ---- transit (\d+)->(\d+)", ln)
        if m:
            tr = int(m.group(1))
            continue
        m = re.match(r"\s*; (PTP|LIN) (\w+)$", ln)
        if m:
            label = m.group(2)
            continue
        if re.match(r"\s*; segment", ln):
            tr, label = None, None
        if i in by_line and label is not None and tr is not None:
            out.append((tr, label, by_line[i]))
            label = None
    return out


def test_results_agree_with_export_reports(results_reports):
    compared = 0
    for f, rep in results_reports.items():
        er = f.parent / "export_report.json"
        if not er.exists():
            continue
        ex = json.loads(er.read_text())
        assert rep["exit_code"] == (1 if ex["flags"] else 0), f.name
        moves = _transit_moves(f, rep)
        assert len(moves) == sum(len(t["moves"]) for t in ex["transits"]), f.name
        for t in ex["transits"]:
            for tr, label, r in moves:
                if tr != t["from_segment"]:
                    continue
                want = t["move_min_clearance_mm"][label]
                got = r["min_clearance_mm"]
                if label in ("lift", "lower"):
                    # the exporter includes the tip on these LINs, the checker exempts it on every LIN
                    assert got >= want - TOL_CLEARANCE_MM, (f.name, tr, label, got, want)
                else:
                    assert abs(got - want) <= TOL_CLEARANCE_MM, (f.name, tr, label, got, want)
                if label == "transit":
                    assert r["kind"] == "PTP"
                    assert abs(r["speed_mm_s"] - t["peak_tcp_mm_s_model"]) <= TOL_SPEED_MM_S, (f.name, tr)
                compared += 1
    assert compared >= 10       # perimeter: 2 transits x 5 moves


# ---------------------------------------------------------------- b. synthetic programs
def test_clean_program_passes(checker, p0):
    rep = check_text(program(clean_body(p0)), checker=checker)
    assert rep["exit_code"] == 0, rep["flags"] + rep["issues"]
    first, last = rep["motions"][0], rep["motions"][-1]
    assert first["speed_mm_s"] is None and first["speed_source"].startswith("start unknown")
    assert 0 < last["speed_mm_s"] <= 100
    assert all(r["speed_mm_s"] == 12.0 for r in rep["motions"] if r["kind"] == "LIN")
    assert all(r["min_clearance_mm"] >= 0 for r in rep["motions"])


def test_start_axis_checks_first_ptp(p0):
    ck = SrcChecker(start_axis_deg=[Q0[0] - 10, *Q0[1:]])
    rep = check_text(program(clean_body(p0)), checker=ck)
    first = rep["motions"][0]
    # unchecked without --start-axis; with it, a 10 deg A1 swing at 5 % is above 100 mm/s in the model
    assert first["speed_mm_s"] > 100
    assert codes(rep) == {"speed"} and [f["line"] for f in rep["flags"]] == [first["line"]]
    assert rep["exit_code"] == 1


def _expect(checker, text, code, exit_code):
    rep = check_text(text, checker=checker)
    assert codes(rep) == {code}, rep["flags"] + rep["issues"]
    assert rep["exit_code"] == exit_code
    return rep


def test_vel_cp_above_limit(checker, p0):
    rep = _expect(checker, program(clean_body(p0), vel_cp=0.15), "speed", 1)
    assert all(f["code"] == "speed" for f in rep["flags"])


def test_target_below_bed_top(checker, p0):
    x, y, z, zp = p0
    body = [lin(x, y, zp), lin(x + 20, y, checker.cfg.bed()["top_z"] - 2.0), lin(x + 20, y, z)]
    _expect(checker, program(body), "nozzle_height", 1)


def test_target_inside_clamp_box(checker, p0):
    x, y, z, zp = p0
    clamp = next(o for o in checker.cfg.cell["obstacles"] if o["name"] == "clamp_plus_x_plus_y")
    cx, cy = clamp["pad_xy"]
    rep = _expect(checker, program([lin(x, y, 40), lin(cx, cy, 40)]), "clearance", 1)
    assert any("clamp_plus_x_plus_y" in f["message"] for f in rep["flags"])


def test_unreachable_target(checker, p0):
    x, y, z, zp = p0
    _expect(checker, program([lin(x, y, 1200)]), "reach", 1)


def test_lin_needing_configuration_change(checker, p0):
    """Tilting the nozzle by 30 deg while turning it about the vertical: the target is reachable,
    but the branch followed from the start of the LIN leaves the limits on the way."""
    x, y, z, zp = p0
    body = [lin(x + 60, y + 60, zp + 40), lin(x + 60, y + 60, zp + 40, -180, 120, 0)]
    rep = _expect(checker, program(body), "lin_branch", 1)
    assert rep["flags"][0]["line"] == rep["motions"][-1]["line"]
    # the target alone is reachable: as a PTP (configuration change allowed) it is not a reach flag
    body_ptp = [body[0], body[1].replace("LIN", "PTP", 1)]
    assert "reach" not in codes(check_text(program(body_ptp), checker=checker))


def test_unmodelled_tool(checker, p0):
    rep = _expect(checker, program(clean_body(p0), tool=5), "frame_not_modelled", 2)
    assert "frame not modelled" in rep["issues"][0]["message"]


def test_unknown_statement(checker, p0):
    body = clean_body(p0)
    rep = _expect(checker, program(body[:2] + ["$OUT[3] = TRUE"] + body[2:]), "not_interpreted", 1)
    line = rep["issues"][0]["line"]
    assert rep["summary"]["not_interpreted_lines"] == [line]


def test_uninterpretable_motion_is_a_parse_error(checker, p0):
    body = clean_body(p0)
    _expect(checker, program(body[:2] + ["CIRC {X 1, Y 2, Z 3}, {X 4, Y 5, Z 6}"] + body[2:]), "parse_error", 2)


# ---------------------------------------------------------------- c. keep-out file
def test_keep_out_file_turns_clean_into_flag(tmp_path, p0):
    x, y, z, zp = p0
    src = tmp_path / "clean.src"
    src.write_text(program(clean_body(p0)))
    rep, md, js = check_file(src, use_local_keep_out=False)
    assert rep["exit_code"] == 0 and md.parent == tmp_path and js.exists()
    ko = tmp_path / "ko.yaml"
    ko.write_text(yaml.safe_dump({"keep_out": [{"name": "test_box", "x": [x + 10, x + 30],
                                                "y": [y + 10, y + 30], "z": [-50, 400]}]}))
    rep, md, js = check_file(src, keep_out_path=ko, out=tmp_path / "with_ko")
    assert rep["exit_code"] == 1
    assert codes(rep) == {"clearance"}
    assert any("keep-out test_box" in f["message"] for f in rep["flags"])
    assert rep["keep_out"]["boxes_from_file"] == 1
    text = md.read_text()
    assert "1 box(es) from that file" in text
    assert f"{x + 10:.3f}" not in text and f"{x + 10:g}" not in text   # no keep-out coordinates


def test_keep_out_example_is_valid_and_far_away(checker):
    from kukasim.src_check import load_keep_out
    boxes = load_keep_out(REPO / "config" / "keep_out.example.yaml")
    assert len(boxes) == 1
    ck = SrcChecker(keep_out=boxes)
    assert len(ck.coll.boxes) == len(checker.coll.boxes) + 1


# ---------------------------------------------------------------- d. RSI dialect
RSI_BODY = """DECL INT RET
DECL INT CONTID
DECL INT OV_SAVE
DECL REAL EXTR_RPM
RET = RSI_CREATE("print_rsi.rsix", CONTID, TRUE)
IF (RET <> RSIOK) THEN
  HALT
ENDIF
OV_SAVE = $OV_PRO
$OV_PRO = 60
EXTR_RPM = 120.0
RET = RSI_ON(#RELATIVE)
IF (RET <> RSIOK) THEN
  HALT
ENDIF
WAIT SEC 0.5
{lins}
RET = RSI_OFF()
EXTR_RPM = 0
$OV_PRO = OV_SAVE
WAIT SEC 0.2
{last}"""


def test_rsi_dialect_program(checker, p0):
    body = clean_body(p0)
    text = program(RSI_BODY.format(lins="\n".join(body[:5]), last="\n".join(body[5:])).splitlines())
    rep = check_text(text, checker=checker)
    assert rep["exit_code"] == 0, rep["flags"] + rep["issues"]
    lines = text.splitlines()
    want = {i for i, ln in enumerate(lines, start=1)
            if re.match(r"(DECL|RET =|IF|  HALT|ENDIF|OV_SAVE|\$OV_PRO|EXTR_RPM|WAIT SEC|DEF|END$)", ln)}
    got = {s["line"] for s in rep["statements"] if s["kind"] == "no_motion"}
    assert got == want
    assert not [s for s in rep["statements"] if s["kind"] == "not_interpreted"]
    kinds = {s["line"]: s["kind"] for s in rep["statements"]}
    assert all(kinds[i] in ("motion", "setting", "frame") for i in range(1, len(lines) + 1) if i not in want)
    in_rsi = [r for r in rep["motions"] if any(n.startswith("RSI on") for n in r["notes"])]
    assert len(in_rsi) == 5


# ---------------------------------------------------------------- parser / output details
def test_parse_variants():
    prog = parse_program(
        "&ACCESS RVP\nDEF x( )\n$TOOL = TOOL_DATA[3]\n$BASE = $NULLFRAME\n$ORI_TYPE = #CONSTANT\n"
        "PTP {E6AXIS: A1 0,A2 -90,A3 90,A4 0,A5 45,A6 0,E1 0,E2 0,E3 0,E4 0} C_PTP\n"
        "lin {X 700,Y -400,Z 50,A 0,B 90,C 0,S 'B110',T 'B110010'} c_dis ; comment\n"
        "PTP {X 700, Y -400, Z 60, A 0, B 90, C 0, S 6, T 50}\nEND\n")
    assert [s.kind for s in prog.statements] == ["no_motion", "no_motion", "frame", "frame", "setting",
                                                 "motion", "motion", "motion", "no_motion"]
    m = prog.motions
    assert m[0].target_type == "axis" and m[0].external["E4"] == 0 and m[0].approx == "C_PTP"
    assert m[1].s == 6 and m[1].t == 50 and m[1].approx == "C_DIS" and m[1].state["ori_type"] == "#CONSTANT"
    assert not prog.issues


def test_never_writes_into_config_or_results(tmp_path):
    d = output_dir(RESULTS / "dome" / "dome.src")
    assert RESULTS not in d.parents and d != RESULTS
    with pytest.raises(ValueError):
        output_dir(tmp_path / "x.src", out=REPO / "config")
    with pytest.raises(ValueError):
        output_dir(tmp_path / "x.src", out=RESULTS / "dome")
    assert output_dir(tmp_path / "x.src") == tmp_path


def test_cli_exit_codes(tmp_path, p0):
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_src", REPO / "scripts" / "check_src.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    ok = tmp_path / "ok.src"
    ok.write_text(program(clean_body(p0)))
    fast = tmp_path / "fast.src"
    fast.write_text(program(clean_body(p0), vel_cp=0.2))
    bad = tmp_path / "bad.src"
    bad.write_text(program(clean_body(p0), tool=1))
    assert cli.main([str(ok), "--no-local-keep-out"]) == 0
    assert cli.main([str(fast), "--no-local-keep-out"]) == 1
    assert cli.main([str(bad), "--no-local-keep-out"]) == 2
    assert (tmp_path / "ok_check.md").exists() and (tmp_path / "ok_check.json").exists()
