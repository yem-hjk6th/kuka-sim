"""Static checker of KRL programs (kukasim/src_check.py, scripts/check_src.py).

Synthetic programs are written here; they hold no site data. Positions are relative to
the approach pose of the committed Hilbert corner program (a point over the bed).
"""
import csv
import json
import math
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
# Joint values against the exporter's trajectory.csv (written with export_report.json): the same 0.05 mm
# at the tool, expressed in joint space. A joint error d moves the TCP by at most d x r, r < 1.5 m (the
# largest TCP distance from any joint axis in the reachable bed area), so 0.05 mm / 1500 mm = 0.0019 deg.
# It covers the csv rounding (0.0001 deg) and the .src target rounding (observed difference <= 0.0006 deg).
TOL_JOINT_DEG = math.degrees(TOL_CLEARANCE_MM / 1500.0)

Q0 = (55.6749, -36.0731, 108.0322, 57.0073, -79.9426, -15.0555)   # results/hilbert_corner approach PTP
Q0_SRC = tuple(float("%g" % v) for v in Q0)     # Q0 as program() writes it: the exact start of its first PTP


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
        # only the first PTP (from the unknown current pose) is not checked
        assert rep["summary"]["not_checked_lines"] == [rep["motions"][0]["line"]], f.name
        assert rep["motions"][0]["kind"] == "PTP" and rep["exit_code"] in (1, 3), f.name
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


def _trajectory(folder):
    """Exporter joints per printed point: {(X, Y, Z) rounded to 0.001 mm: A1..A6 deg}."""
    out = {}
    with open(folder / "trajectory.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["A1"] == "":
                continue
            out[tuple(round(float(row[k]), 3) for k in "XYZ")] = np.array([float(row[f"A{i}"]) for i in range(1, 7)])
    return out


EXPORTED = sorted(f for f in SRC_FILES if (f.parent / "export_report.json").exists())


def test_results_agree_with_export_reports(results_reports):
    """Every committed program with an export_report.json: exit code, transit clearances and PTP speeds
    (cruise bound and ramp-limited peak), the $VEL_AXIS / $ACC_AXIS of the export and the joint values
    of every printed point (trajectory.csv, written by the same export)."""
    assert len(EXPORTED) >= 5
    compared = {}
    for f in EXPORTED:
        rep = results_reports[f]
        ex = json.loads((f.parent / "export_report.json").read_text(encoding="utf-8"))
        n = 0
        assert rep["exit_code"] == (1 if ex["flags"] else 3), f.name        # 3: first PTP not checked
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
                    assert abs(r["speed_peak_ramp_mm_s"] - t["peak_tcp_mm_s_model"]) <= TOL_SPEED_MM_S, (f.name, tr)
                    assert abs(r["speed_mm_s"] - t["ptp_cruise_tcp_mm_s_model"]) <= TOL_SPEED_MM_S, (f.name, tr)
                n += 1
        # speed settings of the export, as the checker reads them at every PTP
        prog = parse_program(f.read_text(encoding="utf-8"))
        first = prog.motions[0]
        assert first.state["vel_axis_pct"] == [ex["params"]["vel_axis_pct"]] * 6, f.name
        assert first.state["acc_axis_pct"] == [ex["params"]["acc_axis_pct"]] * 6, f.name
        # joints of every printed point
        traj = _trajectory(f.parent)
        n_pts = sum(b - a + 1 for a, b in ex["segments"])
        hit = 0
        for r in rep["motions"]:
            key = tuple(round(r["target"][k], 3) for k in "XYZ") if r["target_type"] == "cartesian" else None
            if key in traj and r["kind"] == "LIN":
                assert np.abs(np.array(r["end_axis_deg"]) - traj[key]).max() <= TOL_JOINT_DEG, (f.name, r["line"])
                hit += 1
        assert hit >= n_pts, (f.name, hit, n_pts)
        compared[f.parent.name] = n + hit
    assert compared["perimeter"] >= 10 + 3   # 2 transits x 5 moves, plus the printed points
    assert all(v > 0 for v in compared.values())


# ---------------------------------------------------------------- b. synthetic programs
def test_clean_program_passes(checker, p0):
    rep = check_text(program(clean_body(p0)), checker=checker)
    # no flag; 3 because the first PTP starts from the unknown current pose
    assert rep["exit_code"] == 3, rep["flags"] + rep["issues"]
    first, last = rep["motions"][0], rep["motions"][-1]
    assert first["speed_mm_s"] is None and first["speed_source"].startswith("start unknown")
    assert rep["summary"]["not_checked_lines"] == [first["line"]] and not first["checked"]
    assert 0 < last["speed_mm_s"] <= 100
    assert all(r["speed_mm_s"] == 12.0 for r in rep["motions"] if r["kind"] == "LIN")
    assert all(r["min_clearance_mm"] >= 0 for r in rep["motions"])
    # with the start pose given, every motion is checked: exit 0
    rep0 = check_text(program(clean_body(p0)), checker=SrcChecker(start_axis_deg=Q0_SRC))
    assert rep0["exit_code"] == 0 and rep0["summary"]["not_checked_motions"] == 0, rep0["flags"]


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
    """Tilting the nozzle by 30 deg while turning it by 160 deg about the vertical: the target is
    reachable, but the branch followed from the start of the LIN changes configuration on the way.
    (A 180 deg turn has no unique shortest rotation: see test_lin_reorientation_near_180_not_checked.)"""
    x, y, z, zp = p0
    body = [lin(x + 60, y + 60, zp + 40), lin(x + 60, y + 60, zp + 40, 160, 120, 0)]
    rep = _expect(checker, program(body), "lin_branch", 1)
    assert rep["flags"][0]["line"] == rep["motions"][-1]["line"]
    # the target alone is reachable: as a PTP (configuration change allowed) it is not a reach flag
    body_ptp = [body[0], body[1].replace("LIN", "PTP", 1)]
    assert "reach" not in codes(check_text(program(body_ptp), checker=checker))


def test_lin_reorientation_near_180_not_checked(checker, p0):
    """A #VAR LIN that turns the tool by ~180 deg has no unique shortest rotation (the path would depend
    on rounding): its path is not checked (exit 3), the target pose is."""
    x, y, z, zp = p0
    body = [lin(x + 60, y + 60, zp + 40), lin(x + 60, y + 60, zp + 40, -180, 120, 0)]
    rep = check_text(program(body), checker=checker)
    r = rep["motions"][-1]
    assert not r["checked"] and any("180" in n for n in r["not_checked"])
    assert rep["exit_code"] == 3 and not rep["flags"]


def test_unmodelled_tool(checker, p0):
    rep = _expect(checker, program(clean_body(p0), tool=5), "frame_not_modelled", 2)
    assert "frame not modelled" in rep["issues"][0]["message"]


def test_unknown_statement(checker, p0):
    body = clean_body(p0)
    rep = _expect(checker, program(body[:2] + ["$OUT[3] = TRUE"] + body[2:]), "not_interpreted", 2)
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
    assert rep["exit_code"] == 3 and not rep["flags"] and md.parent == tmp_path and js.exists()
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
    assert rep["exit_code"] == 3 and rep["summary"]["not_checked_motions"] == 1, rep["flags"] + rep["issues"]
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
    assert cli.main([str(ok), "--no-local-keep-out"]) == 3        # first PTP not checked
    assert cli.main([str(ok), "--no-local-keep-out", "--start-axis", *map(str, Q0_SRC)]) == 0
    assert cli.main([str(fast), "--no-local-keep-out"]) == 1
    assert cli.main([str(bad), "--no-local-keep-out"]) == 2
    assert (tmp_path / "ok_check.md").exists() and (tmp_path / "ok_check.json").exists()


# ---------------------------------------------------------------- review items 1-6
def _motion_at(rep, line):
    return next(r for r in rep["motions"] if r["line"] == line)


def _line_of(text, needle, nth=0):
    hits = [i for i, ln in enumerate(text.splitlines(), start=1) if ln.strip() == needle]
    return hits[nth]


@pytest.mark.parametrize("stmt", ["ret = RSI_MOVECORR()", "ret = MyMoveFct(1)", "MySubroutine()",
                                  "MySubroutine(1, 2)", "x = ABS(y)", "IF MyFct() THEN"])
def test_item1_calls_are_never_no_motion(checker, p0, stmt):
    """1. Only an explicit list is no-motion; any call of a function or subroutine is not interpreted
    (exit 2), and the next motion is not checked because the call may have moved the robot."""
    body = clean_body(p0)
    extra = [stmt] + (["ENDIF"] if stmt.startswith("IF") else [])
    text = program(body[:2] + extra + body[2:])
    rep = check_text(text, checker=checker)
    ln = _line_of(text, stmt)
    kinds = {s["line"]: s["kind"] for s in rep["statements"]}
    assert kinds[ln] == "not_interpreted"
    assert rep["exit_code"] == 2 and "not_interpreted" in codes(rep)
    nxt = next(r for r in rep["motions"] if r["line"] > ln)
    assert not nxt["checked"] and any("not interpreted may have moved" in n for n in nxt["not_checked"])


def test_item1_allow_list_still_accepted(checker, p0):
    body = clean_body(p0)
    extra = ["DECL REAL X", "X = 3", "X = Y + 2 * (Z - 1)", "X = $OV_PRO", "WAIT SEC 0.2",
             "IF (X <> 3) AND NOT (Y > 2) THEN", "HALT", "ENDIF"]
    rep = check_text(program(body[:2] + extra + body[2:]), checker=checker)
    assert rep["exit_code"] == 3 and not rep["issues"], rep["issues"]
    assert rep["summary"]["not_checked_motions"] == 1           # the first PTP only


def test_item2_if_branch_uses_worst_case(checker, p0):
    """2. $VEL.CP = 0.5, then IF x THEN $VEL.CP = 0.01 ENDIF: the LIN gets an M1 flag (500 mm/s)."""
    x, y, z, zp = p0
    body = ["$VEL.CP = 0.5", "IF TESTFLAG THEN", "$VEL.CP = 0.01", "ENDIF", lin(x, y, zp)]
    text = program(body)
    rep = check_text(text, checker=checker)
    r = rep["motions"][-1]
    assert r["speed_mm_s"] == 500.0 and any(f["code"] == "speed" for f in r["flags"])
    assert rep["exit_code"] == 1


def test_item2_if_else_and_switch(checker, p0):
    x, y, z, zp = p0
    body = ["IF TESTFLAG THEN", "$VEL.CP = 0.01", "ELSE", "$VEL.CP = 0.2", "ENDIF", lin(x, y, zp)]
    r = check_text(program(body), checker=checker)["motions"][-1]
    assert r["speed_mm_s"] == 200.0 and [f["code"] for f in r["flags"]] == ["speed"]
    # both branches set the same value: certain, no flag
    same = ["IF TESTFLAG THEN", "$VEL.CP = 0.01", "ELSE", "$VEL.CP = 0.01", "ENDIF", lin(x, y, zp)]
    r = check_text(program(same), checker=checker)["motions"][-1]
    assert r["speed_mm_s"] == 10.0 and not r["flags"]
    # SWITCH without DEFAULT: the incoming 150 mm/s may stay in force
    sw = ["$VEL.CP = 0.15", "SWITCH MODE", "CASE 1", "$VEL.CP = 0.01", "CASE 2", "$VEL.CP = 0.012",
          "ENDSWITCH", lin(x, y, zp)]
    r = check_text(program(sw), checker=checker)["motions"][-1]
    assert r["speed_mm_s"] == 150.0 and r["flags"]
    # a frame that may be active: TOOL 5 in one branch -> the LIN after it is not checked
    tb = ["IF TESTFLAG THEN", "BAS(#TOOL,5)", "ENDIF", lin(x, y, zp)]
    rep = check_text(program(tb), checker=checker)
    assert not rep["motions"][-1]["checked"] and rep["exit_code"] == 2


def test_item2_loop_reaches_every_state(checker, p0):
    """Motions in a loop body are checked for every setting state the loop can reach: the first LIN of the
    body runs at 150 mm/s from the second iteration on."""
    x, y, z, zp = p0
    body = [lin(x, y, zp), "FOR I = 1 TO 3", lin(x + 10, y, zp), "$VEL.CP = 0.15", lin(x + 20, y, zp),
            "ENDFOR", lin(x, y, zp)]
    text = program(body)
    rep = check_text(text, checker=checker)
    ln = [i for i, t in enumerate(text.splitlines(), start=1) if t.startswith("LIN")]
    first_in, second_in, after = (_motion_at(rep, n) for n in ln[1:])
    assert first_in["speed_mm_s"] == 150.0 and any(f["code"] == "speed" for f in first_in["flags"])
    assert any("loop" in n for n in first_in["not_checked"])     # start pose depends on the loop
    assert second_in["speed_mm_s"] == 150.0 and second_in["flags"]
    assert after["speed_mm_s"] == 150.0 and after["flags"]       # the loop may run: 150 mm/s after it
    # a WHILE loop may not run at all: the setting before it stays possible
    wl = ["$VEL.CP = 0.15", "WHILE TESTFLAG", "$VEL.CP = 0.01", "ENDWHILE", lin(x, y, zp)]
    assert check_text(program(wl), checker=checker)["motions"][-1]["flags"]


def test_item3_ptp_flag_on_cruise_bound(p0):
    """3. A short PTP whose ramp-limited peak stays below 100 mm/s is flagged on its cruise bound."""
    ck = SrcChecker(start_axis_deg=Q0_SRC)
    body = ["BAS(#VEL_PTP,5)", "PTP {A1 %g, A2 %g, A3 %g, A4 %g, A5 %g, A6 %g}" % (Q0[0] + 0.5, *Q0[1:])]
    rep = check_text(program(body), checker=ck)
    r = rep["motions"][-1]
    assert r["speed_peak_ramp_mm_s"] < 100 < r["speed_mm_s"]
    assert [f["code"] for f in r["flags"]] == ["speed"] and "cruise bound" in r["flags"][0]["message"]
    assert "cruise bound (flag value)" in r["speed_source"] and "ASSUMED axis accelerations" in r["speed_source"]


def test_item4_unknown_state_after_statement_not_interpreted(p0):
    """4. $TOOL = $NULLFRAME and BAS(#VEL_CP, ...) make TOOL / $VEL.CP unknown: later motions are not
    checked until a modelled statement sets them again."""
    ck = SrcChecker()
    x, y, z, zp = p0
    ptp_q0 = "PTP {A1 %g, A2 %g, A3 %g, A4 %g, A5 %g, A6 %g}" % Q0
    body = [lin(x, y, zp), "$TOOL = $NULLFRAME", lin(x + 10, y, zp), lin(x + 20, y, zp),
            "$TOOL = TOOL_DATA[3]", ptp_q0, lin(x, y, zp),
            "BAS(#VEL_CP, 0.5)", "$TOOL = TOOL_DATA[3]", "$BASE = $NULLFRAME", ptp_q0, lin(x + 10, y, zp),
            "$VEL.CP = 0.012", "BAS(#VEL_PTP,5)", ptp_q0, lin(x + 20, y, zp)]
    rep = check_text(program(body), checker=ck)
    assert rep["exit_code"] == 2
    ms = rep["motions"][1:]          # after the first PTP
    assert ms[0]["checked"]
    assert any("$TOOL unknown" in n for n in ms[1]["not_checked"])
    assert any("$TOOL unknown" in n for n in ms[2]["not_checked"])
    assert ms[3]["kind"] == "PTP" and not ms[3]["checked"]       # start after an unchecked LIN is unknown
    assert ms[4]["checked"], ms[4]["not_checked"]                # TOOL set again, start known
    # BAS(#VEL_CP, ...) is a call: every setting and the pose are unknown after it
    assert ms[5]["kind"] == "PTP" and any("$VEL_AXIS[1]" in n and "unknown" in n for n in ms[5]["not_checked"])
    assert any("$VEL.CP unknown" in n for n in ms[6]["not_checked"])
    assert ms[7]["checked"] and ms[8]["checked"] and ms[8]["speed_mm_s"] == 12.0


def test_item5_first_cartesian_motion_not_checked(checker, p0):
    """5. A first Cartesian motion from an unknown start is not checked; exit 3; counted in the summary."""
    x, y, z, zp = p0
    text = program([lin(x, y, zp), lin(x + 10, y, zp)]).replace(
        "PTP {A1 %g, A2 %g, A3 %g, A4 %g, A5 %g, A6 %g}\n" % Q0, "")
    rep = check_text(text, checker=checker)
    first = rep["motions"][0]
    assert first["kind"] == "LIN" and not first["checked"]
    assert first["not_checked"][0].startswith("start unknown")
    assert rep["summary"]["not_checked_motions"] == 1 and rep["exit_code"] == 3
    assert rep["result"].startswith("no flag, but")
    # a flag still wins over "not checked"
    assert check_text(program([lin(x, y, zp)], vel_cp=0.2), checker=checker)["exit_code"] == 1


def test_item6_pdat_and_no_effect_settings(p0):
    """6. PDAT_ACT.VEL / ACC feed BAS(#PTP_PARAMS); APO_DIST, $BWDSTART, $ADVANCE are listed."""
    ck = SrcChecker(start_axis_deg=Q0_SRC)
    tgt = "PTP {A1 %g, A2 %g, A3 %g, A4 %g, A5 %g, A6 %g}" % (Q0[0] + 5, *Q0[1:])
    extra = ["PDAT_ACT.VEL = 50", "PDAT_ACT.ACC = 20", "PDAT_ACT.APO_DIST = 10", "$BWDSTART = FALSE",
             "$ADVANCE = 3", "BAS(#PTP_PARAMS, 1)"]
    text = program(extra + [tgt])
    prog = parse_program(text)
    m = prog.motions[-1]
    assert m.state["vel_axis_pct"] == [50.0] * 6 and m.state["acc_axis_pct"] == [20.0] * 6
    rep = check_text(text, checker=ck)
    assert rep["motions"][-1]["flags"] and rep["exit_code"] == 1           # 50 % of the axis speed
    no_eff = rep["summary"]["no_effect_setting_lines"]
    assert no_eff == [_line_of(text, s) for s in extra[2:5]]
    slow = program(["PDAT_ACT.VEL = 1", "PDAT_ACT.ACC = 10", "BAS(#PTP_PARAMS, 1)", tgt])
    rep = check_text(slow, checker=ck)
    assert rep["exit_code"] == 0 and rep["motions"][-1]["speed_mm_s"] < 100
    # PDAT_ACT from a data set (not read): $VEL_AXIS unknown after BAS(#PTP_PARAMS) -> not checked
    ds = program(["PDAT_ACT = PPDAT1", "BAS(#PTP_PARAMS, 100)", tgt])
    rep = check_text(ds, checker=ck)
    assert rep["exit_code"] == 3 and not rep["motions"][-1]["checked"]
