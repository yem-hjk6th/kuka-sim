"""Static safety check of a KRL program (.src) against the cell model. Simulation only.

    python scripts/check_src.py results/hilbert_corner/hilbert_corner.src
    python scripts/check_src.py my_program.src --out out/check --keep-out my_keep_out.yaml
    python scripts/check_src.py my_program.src --start-axis 0 -90 90 0 45 0

Writes <program>_check.md and <program>_check.json next to the program, or into --out
(a program under config/ or results/ is reported into out/src_check/ instead: the checker
never writes into config/ or results/). Exit code: 0 no flag, 1 at least one flag,
2 parse or frame error. "No flag" means no flag in this model; it does not replace a
check on the controller.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kukasim.src_check import check_file  # noqa: E402


def _rel(p):
    try:
        return Path(p).resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("program", help="KRL .src file")
    ap.add_argument("--out", help="output directory (default: next to the program)")
    ap.add_argument("--keep-out", help="YAML file with extra keep-out boxes (cell.yaml keep_out format); "
                                       "replaces config/keep_out.local.yaml")
    ap.add_argument("--no-local-keep-out", action="store_true",
                    help="do not load config/keep_out.local.yaml")
    ap.add_argument("--start-axis", nargs=6, type=float, metavar=("A1", "A2", "A3", "A4", "A5", "A6"),
                    help="joint values (deg) before the first motion; otherwise the first PTP is not checked")
    a = ap.parse_args(argv)
    try:
        rep, md, js = check_file(a.program, out=a.out, keep_out_path=a.keep_out,
                                 use_local_keep_out=not a.no_local_keep_out, start_axis_deg=a.start_axis)
    except (OSError, ValueError) as e:
        print(f"check_src: {e}", file=sys.stderr)
        return 2
    s = rep["summary"]
    print(f"{rep['program']}: {rep['result']} (exit {rep['exit_code']}); {s['motions']} motions, "
          f"{s['flagged_motions']} flagged; flags: "
          + (", ".join(f"{k} {v}" for k, v in s["flag_counts"].items()) or "none")
          + (f"; not interpreted: lines {s['not_interpreted_lines']}" if s["not_interpreted_lines"] else ""))
    for i in rep["issues"]:
        if i["severity"] == "error":
            print(f"  error, line {i['line']}: {i['message']}")
    print(f"  keep-out: {rep['keep_out']['source']}, {rep['keep_out']['boxes_from_file']} box(es)")
    print(f"  report: {_rel(md)}\n  json:   {_rel(js)}")
    return rep["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
