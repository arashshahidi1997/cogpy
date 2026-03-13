#!/usr/bin/env python3
"""CLI entry point for Snakemake-based ECoG preprocessing."""
import textwrap
import re
import yaml
from importlib.resources import files, as_file
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# optional: only needed for --install-completion
try:
    import shtab  # pip install shtab
except Exception:
    shtab = None

# optional tab-completion support
try:
    import argcomplete  # type: ignore
except Exception:
    argcomplete = None


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    pass


# Step -> output extension (must match workflow rules)
STEP_EXT = {
    "raw_zarr": ".zarr",
    "lowpass": ".zarr",
    "downsample": ".zarr",
    "feature": ".zarr",
    "badlabel": ".npy",
    "plot_feature_maps": ".png",
    "interpolate": ".zarr",
    "all": ".all",
}
VALID_STEPS = tuple(STEP_EXT.keys())

# Simple styling helpers (colors only if output is a TTY)
ANSI = sys.stdout.isatty()
BOLD = "\033[1m" if ANSI else ""
YELLOW = "\033[33m" if ANSI else ""
RESET = "\033[0m" if ANSI else ""


def box_line(content: str, width: int) -> str:
    """Pad content (ignoring ANSI escapes) to fit box width."""
    # Strip ANSI escape codes for length calculation
    visible = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", content)
    padding = width - len(visible)
    return f"{YELLOW}│{RESET}{content}{' ' * padding}{YELLOW}│{RESET}"


QUICK = f"{BOLD}$ cogpy-preproc all path/to/file/filename.lfp{RESET}"
WIDTH = 63  # visible width inside box

BOXED_QUICK = (
    f"{YELLOW}┌{'─' * WIDTH}┐{RESET}\n"
    f"{box_line(' Quick start (most common):', WIDTH)}\n"
    f"{box_line('   ' + QUICK, WIDTH)}\n"
    f"{YELLOW}└{'─' * WIDTH}┘{RESET}"
)

DESC = f"""\
Detects and interpolates bad channels for a single pipeline step.

INPUT
  input_spec  = data/../filepath.lfp
  example     = data/devA/devA-S01/rec1.lfp

STEPS
  {", ".join(VALID_STEPS)}

TARGET (built by the program)
  preproc-results/<step>/<filepath>.<ext>
"""

EXAMPLES = f"""\
{BOXED_QUICK}

Other examples:
  # Run lowpass with 8 cores and print shell commands
  cogpy-preproc lowpass data/devA/devA-S01/rec1 -c 8 --printshellcmds

  # Plan (dry-run) feature extraction without executing
  cogpy-preproc feature data/devA/devA-S01/rec1 --dry-run

  # Merge user overrides into packaged defaults
  cogpy-preproc interpolate data/devA/devA-S01/rec1 --configfile my-overrides.yml
"""


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (mutates base)."""
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _load_default_config() -> dict:
    with as_file(files("cogpy.workflows.preprocess") / "config.yml") as cfg_path:
        data = Path(cfg_path).read_text()
    return yaml.safe_load(data) or {}


def _build_completion_parser(prog: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=prog, formatter_class=_HelpFormatter, add_help=False
    )

    # Positional args
    step = ap.add_argument("step", choices=VALID_STEPS, help="Pipeline step")
    spec = ap.add_argument("input_spec", help="Input spec: path/to/file/filename.lfp")

    # Options
    ap.add_argument("-c", "--cores", type=int, default=8)
    ap.add_argument("-n", "--dry-run", action="store_true")
    ap.add_argument("--printshellcmds", action="store_true")
    cfg = ap.add_argument("--configfile", type=Path)

    # Attach file/path completers (this is the important part)
    if shtab:
        try:
            spec.complete = shtab.FILE  # positional: input_spec -> files/paths
            cfg.complete = shtab.FILE  # option: --configfile -> files/paths
        except Exception:
            pass

    return ap


def _install_completion_bash(prog: str) -> None:
    """Generate and install bash completion for this CLI."""
    if not shtab:
        sys.exit("shtab is not installed. Run: pip install shtab")

    comp = shtab.complete(_build_completion_parser(prog), shell="bash")

    target_dir = Path.home() / ".bash_completion.d"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / prog
    path.write_text(comp)

    rc = Path.home() / ".bashrc"
    block = textwrap.dedent(
        """
	# load local bash completions
	if [ -d ~/.bash_completion.d ]; then
	  for f in ~/.bash_completion.d/*; do
		[ -r "$f" ] && . "$f"
	  done
	fi
	"""
    ).strip("\n")

    existing = rc.read_text() if rc.exists() else ""
    if ".bash_completion.d/*" not in existing:
        with rc.open("a") as fh:
            fh.write("\n" + block + "\n")

    print(f"Installed bash completion at: {path}")
    print("✅ Open a new terminal or run: source ~/.bashrc")


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]

    # Handle completion install and exit early
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--install-completion",
        action="store_true",
        help="Install bash completion for this command and exit.",
    )
    ns, remaining = pre.parse_known_args(argv)
    if ns.install_completion:
        _install_completion_bash("cogpy-preproc")  # matches your prog name
        return

    ap = argparse.ArgumentParser(
        prog="cogpy-preproc",
        description=DESC,
        epilog=EXAMPLES,
        formatter_class=_HelpFormatter,
    )

    # Positional args
    ap.add_argument("step", choices=VALID_STEPS, help="Pipeline step to build")
    ap.add_argument("input_spec", help="Input spec: path/to/file/filename.lfp")

    # Core options
    ap.add_argument(
        "-c",
        "--cores",
        type=int,
        default=8,
        help="Max cores for Snakemake (-c/--cores)",
    )
    ap.add_argument(
        "-n", "--dry-run", action="store_true", help="Dry-run (plan only, no execution)"
    )
    ap.add_argument(
        "--printshellcmds",
        action="store_true",
        help="Print shell commands executed by Snakemake",
    )
    # Config merging
    ap.add_argument(
        "--configfile",
        type=Path,
        help="User yml to merge with packaged defaults (deep merge)",
    )

    # enable argcomplete if available
    if argcomplete is not None:
        argcomplete.autocomplete(ap)
    args, smk_extra = ap.parse_known_args(remaining)

    # Build target path per your generate_pipe_path()
    workdir = Path(args.input_spec).parent
    filename = Path(args.input_spec).with_suffix(STEP_EXT[args.step]).name
    target = Path("preproc-results") / args.step / filename

    # Prepare merged config: packaged defaults + optional user overrides
    merged_cfg = _load_default_config()
    if args.configfile:
        if not args.configfile.exists():
            sys.exit(f"config file not found: {args.configfile}")
        user_cfg = yaml.safe_load(args.configfile.read_text()) or {}
        _deep_merge(merged_cfg, user_cfg)

    # Resolve packaged Snakefile to a real path and write merged config to a temp file
    with (
        as_file(files("cogpy.workflows.preprocess") / "Snakefile") as snakefile_path,
        tempfile.TemporaryDirectory() as td,
    ):
        merged_cfg_path = Path(td) / "config.merged.yml"
        merged_cfg_path.write_text(yaml.safe_dump(merged_cfg, sort_keys=False))

        cmd = [
            sys.executable,
            "-m",
            "snakemake",
            "-s",
            str(snakefile_path),
            "--configfile",
            str(merged_cfg_path),
            "-d",
            str(workdir),
            "-c",
            str(args.cores),
        ]
        cmd += smk_extra  # pass-through to Snakemake
        cmd.append(target)  # our computed target last
        if args.printshellcmds:
            cmd.append("--printshellcmds")
        if args.dry_run:
            cmd.append("-n")

        print("Running:\n $ ", " ".join(map(str, cmd)))
        raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
