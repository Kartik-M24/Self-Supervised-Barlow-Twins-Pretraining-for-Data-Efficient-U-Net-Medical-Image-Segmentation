"""CPU smoke test for the whole pipeline, on a synthetic dataset.

    python smoke_test.py                       # everything, both datasets
    python smoke_test.py --only notebooks      # just the five training/eval notebooks
    python smoke_test.py --only scripts arch   # just the ablation scripts
    python smoke_test.py --datasets KDSB --keep

Checks that every stage runs end to end. It says nothing about model quality:
images are random blobs, training is one epoch, and the numbers are meaningless.

Everything happens in a throwaway workspace (a temp dir, or --workdir), so the
real datasets/, saved_models/, saved_np/ and results/ are never read or written.
The repository's notebooks are executed from copies with a handful of config
lines rewritten; the originals are not modified.

Stages
  notebooks  UNet-Baseline, BT-UNet, SimSiam-UNet and Dense-BT-UNet with
             TEST_MODE = True at 32x32, per dataset, then Evaluation_Pipeline.
  scripts    Ablations/: pretrain.py, finetune.py (scratch, barlow, simsiam,
             frozen, and a resume re-run), tier0_tables.py, tier0_sweep.py,
             ablation_report.py, eval_pattern_patch.py. The scripts are fixed at
             256x256, so this stage takes the longest.
  arch       Ablations/Arch/: verify_arch.py, plus finetune.py with
             --no-output-bn, --attention and a depth-3 encoder.
  drivers    The setup cells of both ablation driver notebooks, checking that
             PROJECT_DIR / ABLATION_DIR resolve with and without the environment
             variable override.

The GPU is hidden from every child process (CUDA_VISIBLE_DEVICES=-1), so this
runs on the CPU even on a GPU machine. Executed notebooks and logs are written
under <workdir>/executed and <workdir>/logs for inspection.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent          # "BT Unet/"
ABL_DIR = REPO_DIR / "Ablations"
ARCH_DIR = ABL_DIR / "Arch"

NB_IMG = 32          # notebook image size; must be divisible by 16 (four poolings)
SRC_IMG = 64         # size of the synthetic source images on disk
N_TRAIN, N_VAL, N_TEST = 48, 8, 8   # 48 = six full batches of 8 for SSL pretraining
CELL_TIMEOUT = 1800

class Skip(Exception):
    """A stage that cannot run in this environment; reported, not failed."""


CPU_ENV = {"CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "2",
           "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}


# ============================================================
# Synthetic data
# ============================================================

def make_dataset(root, name, rng):
    """Random-blob images in the on-disk layout each dataset's loader expects."""
    from PIL import Image

    if name == "ISIC2018":
        splits = {"train": N_TRAIN, "val": N_VAL, "test": N_TEST}
        img_name = lambda i: f"ISIC_{i:07d}.jpg"
        mask_name = lambda img: img[:-4] + "_segmentation.png"
    else:   # KDSB: no val split, mask shares the image's filename
        splits = {"train": N_TRAIN, "test": N_TEST}
        img_name = lambda i: f"img_{i:04d}.png"
        mask_name = lambda img: img

    yy, xx = np.mgrid[:SRC_IMG, :SRC_IMG]
    i = 0
    for split, n in splits.items():
        org = root / "datasets" / name / split / "org"
        gt = root / "datasets" / name / split / "gt"
        org.mkdir(parents=True, exist_ok=True)
        gt.mkdir(parents=True, exist_ok=True)
        for _ in range(n):
            mask = np.zeros((SRC_IMG, SRC_IMG), bool)
            for _ in range(rng.integers(1, 4)):
                cy, cx = rng.integers(8, SRC_IMG - 8, 2)
                ry, rx = rng.integers(4, 14, 2)
                mask |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
            img = rng.integers(0, 90, (SRC_IMG, SRC_IMG, 3))
            img[mask] += rng.integers(90, 160, 3)
            f = img_name(i)
            Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(org / f)
            Image.fromarray(mask.astype(np.uint8) * 255).save(gt / mask_name(f))
            i += 1


def make_workspace(root, datasets, seed=0):
    rng = np.random.default_rng(seed)
    for name in datasets:
        make_dataset(root, name, rng)
    for d in ("saved_models", "saved_np", "results", "logs/fit"):
        (root / d).mkdir(parents=True, exist_ok=True)


# ============================================================
# Notebook execution
# ============================================================

def install_kernel(workdir):
    """A kernelspec that runs this interpreter, so the notebooks see the same
    packages as this script regardless of what 'python3' resolves to."""
    spec = workdir / "jupyter" / "kernels" / "smoke-python"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "smoke-test python", "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = os.pathsep.join(
        [str(workdir / "jupyter")] + ([os.environ["JUPYTER_PATH"]] if os.environ.get("JUPYTER_PATH") else []))
    return "smoke-python"


def cell_src(c):
    return "".join(c["source"])


def set_config(nb, overrides):
    """Rewrite `NAME = value` lines in the config cell. Each key must match exactly
    one line, so a renamed setting fails here instead of silently not applying."""
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"
             and re.search(r"^# CONFIGURATION", cell_src(c), re.M)]
    assert len(cells) == 1, "expected exactly one CONFIGURATION cell"
    src = cell_src(cells[0])
    for key, value in overrides.items():
        pat = re.compile(rf"^{re.escape(key)}\s*=.*$", re.M)
        n = len(pat.findall(src))
        assert n == 1, f"config key {key!r} matched {n} lines"
        src = pat.sub(lambda _: f"{key} = {value}", src)
    cells[0]["source"] = src


def set_line(nb, key, value):
    """Rewrite a `NAME = value` line wherever it sits (some notebooks define
    RECALL_FLOOR in the K-fold cell rather than the config cell)."""
    pat = re.compile(rf"^{re.escape(key)}\s*=.*$", re.M)
    hits = [c for c in nb["cells"] if c["cell_type"] == "code" and pat.search(cell_src(c))]
    assert len(hits) == 1 and len(pat.findall(cell_src(hits[0]))) == 1, f"{key!r} not found exactly once"
    hits[0]["source"] = pat.sub(lambda _: f"{key} = {value}", cell_src(hits[0]))


def force_cpu(nb):
    """Replace every GPU pin with -1. The pin sits in a notebook cell and would
    otherwise override the environment variable."""
    pat = re.compile(r'os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"[^"]*"')
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["source"] = pat.sub('os.environ["CUDA_VISIBLE_DEVICES"] = "-1"', cell_src(c))


def execute(nb, cwd, out_path, kernel, env=None):
    import nbformat
    from nbclient import NotebookClient

    # reads() rather than from_dict(): it joins list-form cell sources into strings.
    node = nbformat.reads(json.dumps(nb), as_version=4)
    old_env = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    try:
        NotebookClient(node, timeout=CELL_TIMEOUT, kernel_name=kernel,
                       resources={"metadata": {"path": str(cwd)}}).execute()
    finally:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(node, str(out_path))
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return node


def load_nb(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def notebook_stages(ws, datasets, kernel, executed):
    stages = []

    def training(nb_name, dataset, extra):
        def run():
            nb = load_nb(REPO_DIR / nb_name)
            set_config(nb, {"DATASET_NAME": f'"{dataset}"', "IMG_HEIGHT": NB_IMG,
                            "IMG_WIDTH": NB_IMG, "LABEL_FRACTIONS": "[0.50]",
                            "TEST_MODE": True, **extra})
            # One epoch on random blobs often predicts nothing, and the recall
            # floor would then drop every fold and skip the ensemble (and the
            # visualisation cell, which loads it, would fail). Zero keeps the
            # ensemble path deterministic.
            set_line(nb, "RECALL_FLOOR", 0.0)
            force_cpu(nb)
            execute(nb, ws, executed / f"{Path(nb_name).stem}_{dataset}.ipynb", kernel)
            # TEST_MODE trains 2 folds at the first fraction and then ensembles them.
            models = sorted(p.name for p in (ws / "saved_models").glob(f"model_{dataset}_*50pct*ensemble.keras"))
            assert models, "no ensemble model written"
        return run

    ssl = {"PRETRAIN_FLG": True, "PRETRAIN_EPOCHS": 1}
    for ds in datasets:
        stages += [
            (f"notebook  UNet-Baseline   [{ds}]", training("UNet-Baseline.ipynb", ds, {})),
            (f"notebook  BT-UNet         [{ds}]", training("BT-UNet.ipynb", ds, ssl)),
            (f"notebook  SimSiam-UNet    [{ds}]", training("SimSiam-UNet.ipynb", ds, ssl)),
            (f"notebook  Dense-BT-UNet   [{ds}]", training("Dense-BT-UNet.ipynb", ds, {"PRETRAIN_FLG": True})),
        ]

    def evaluation():
        nb = load_nb(REPO_DIR / "Evaluation_Pipeline.ipynb")
        set_config(nb, {"DATASETS_TO_EVALUATE": json.dumps(datasets),
                        "IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS": f"{NB_IMG}, {NB_IMG}, 3",
                        "N_FOLDS": 2})
        force_cpu(nb)
        execute(nb, ws, executed / "Evaluation_Pipeline.ipynb", kernel)
        assert (ws / "results" / "evaluation_results.csv").exists(), "no evaluation_results.csv"

    stages.append(("notebook  Evaluation_Pipeline", evaluation))
    return stages


# ============================================================
# Ablation scripts
# ============================================================

def run_script(script, args, cwd, log_dir, pythonpath, expect=None):
    env = {**os.environ, **CPU_ENV,
           "PYTHONPATH": os.pathsep.join(str(p) for p in pythonpath)}
    cmd = [sys.executable, "-u", str(script), *[str(a) for a in args]]
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{script.stem}_{int(time.time() * 1000)}.log"
    log.write_text("$ " + " ".join(cmd) + "\n\n" + proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:])
        raise RuntimeError(f"{script.name} exited {proc.returncode} (log: {log})\n{tail}")
    if expect and expect not in proc.stdout:
        raise AssertionError(f"{script.name}: expected {expect!r} in output (log: {log})")
    return proc.stdout


def script_stages(ws, datasets, logs):
    parent = lambda name, args, **kw: run_script(ABL_DIR / name, args, ws, logs, [ABL_DIR], **kw)
    ft = ["--gpu", "-1", "--fractions", "0.25", "--folds", "2", "--epochs", "1", "--fast-metrics"]

    stages = [("script    eval_pattern_patch.py", lambda: parent("eval_pattern_patch.py", []))]
    for ds in datasets:
        def pretrain(ds=ds):
            for m in ("barlow", "simsiam"):
                parent("pretrain.py", ["--method", m, "--dataset", ds, "--gpu", "-1", "--epochs", "1"])

        def finetune(ds=ds):
            parent("finetune.py", ["--dataset", ds, "--scratch", *ft])
            for m in ("barlow", "simsiam"):
                parent("finetune.py", ["--dataset", ds, "--method", m, *ft])
            parent("finetune.py", ["--dataset", ds, "--method", "barlow", "--freeze-encoder",
                                   "--variant", "frozen", *ft])

        def resume(ds=ds):
            # A second identical run must skip every finished fold.
            parent("finetune.py", ["--dataset", ds, "--method", "barlow", *ft],
                   expect="Complete, skipping.")

        def tables():
            import pandas as pd
            if tuple(int(x) for x in pd.__version__.split(".")[:2]) < (2, 2):
                raise Skip(f"tier0_tables.py needs pandas>=2.2 "
                           f"(groupby.apply(include_groups=...)); found {pd.__version__}")
            parent("tier0_tables.py", ["--results-dir", "results", "--out-dir", "results/tier0"])

        stages += [
            (f"script    pretrain.py barlow+simsiam       [{ds}]", pretrain),
            (f"script    finetune.py scratch/bt/ss/frozen [{ds}]", finetune),
            (f"script    finetune.py resume               [{ds}]", resume),
            (f"script    tier0_tables.py                  [{ds}]", tables),
            (f"script    tier0_sweep.py                   [{ds}]",
             lambda ds=ds: parent("tier0_sweep.py", ["--dataset", ds, "--gpu", "-1", "--tune-on-foldval"])),
            (f"script    ablation_report.py               [{ds}]",
             lambda ds=ds: parent("ablation_report.py", ["--results-dir", "results", "--dataset", ds])),
        ]
    return stages


def arch_stages(ws, datasets, logs):
    arch = lambda name, args: run_script(ARCH_DIR / name, args, ws, logs, [ARCH_DIR, ABL_DIR])
    ds = datasets[0]
    ft = ["--gpu", "-1", "--fractions", "0.25", "--folds", "2", "--epochs", "1", "--fast-metrics"]

    def variants():
        arch("finetune.py", ["--dataset", ds, "--scratch", "--no-output-bn", "--variant", "outbn0", *ft])
        if not (ws / "saved_models" / f"{ds}_barlow_twins_encoder.keras").exists():
            arch("pretrain.py", ["--method", "barlow", "--dataset", ds, "--gpu", "-1", "--epochs", "1"])
        arch("finetune.py", ["--dataset", ds, "--method", "barlow", "--attention", "--variant", "attn", *ft])

    def depth3():
        arch("pretrain.py", ["--method", "barlow", "--dataset", ds, "--gpu", "-1", "--epochs", "1",
                             "--depth", "3", "--variant", "d3"])
        arch("finetune.py", ["--dataset", ds, "--method", "barlow", "--encoder-variant", "d3",
                             "--variant", "d3", *ft])

    return [
        ("arch      verify_arch.py", lambda: arch("verify_arch.py", [])),
        (f"arch      finetune --no-output-bn / --attention [{ds}]", variants),
        (f"arch      pretrain+finetune --depth 3 [{ds}]", depth3),
    ]


# ============================================================
# Driver notebook path resolution
# ============================================================

def driver_stages(ws, kernel, executed):
    def trimmed(path, stop_marker, check):
        nb = load_nb(path)
        idx = next(i for i, c in enumerate(nb["cells"]) if stop_marker in cell_src(c))
        nb["cells"] = nb["cells"][: idx + 1]
        nb["cells"].append({"cell_type": "code", "execution_count": None, "id": "smoke-check",
                            "metadata": {}, "outputs": [], "source": check})
        return nb

    def main_driver():
        path = ABL_DIR / "Ablation_Driver.ipynb"
        # Default: PROJECT_DIR is the folder above Ablations/.
        nb = trimmed(path, "def run(", f"assert PROJECT_DIR == Path(r'{REPO_DIR}').resolve(), PROJECT_DIR\n"
                                        f"assert ABLATION_DIR == Path(r'{ABL_DIR}').resolve(), ABLATION_DIR\n")
        execute(nb, ABL_DIR, executed / "Ablation_Driver_default.ipynb", kernel)
        # Override: data somewhere else; also runs a script through the run() helper.
        nb = trimmed(path, 'run("eval_pattern_patch.py")',
                     f"assert PROJECT_DIR == Path(r'{ws}').resolve(), PROJECT_DIR\n"
                     f"assert Path.cwd().resolve() == PROJECT_DIR\n"
                     f"assert missing == [], missing\n")
        execute(nb, ABL_DIR, executed / "Ablation_Driver_override.ipynb", kernel,
                env={"PROJECT_DIR": str(ws)})

    def arch_driver():
        path = ARCH_DIR / "Ablation_Arch_Driver.ipynb"
        nb = trimmed(path, "def run(", f"assert PROJECT_DIR == Path(r'{REPO_DIR}').resolve(), PROJECT_DIR\n"
                                       f"assert PARENT_DIR == Path(r'{ABL_DIR}').resolve(), PARENT_DIR\n"
                                       f"assert ABLATION_DIR == Path(r'{ARCH_DIR}').resolve(), ABLATION_DIR\n"
                                       f"assert not missing_arch and not missing_parent\n"
                                       f"assert run('eval_pattern_patch.py') == 0\n")
        execute(nb, ARCH_DIR, executed / "Ablation_Arch_Driver_default.ipynb", kernel)

    return [("driver    Ablation_Driver setup (default + PROJECT_DIR override)", main_driver),
            ("driver    Ablation_Arch_Driver setup", arch_driver)]


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=["notebooks", "scripts", "arch", "drivers"],
                   default=["notebooks", "scripts", "arch", "drivers"])
    p.add_argument("--datasets", nargs="+", choices=["ISIC2018", "KDSB"], default=["ISIC2018", "KDSB"])
    p.add_argument("--workdir", type=Path, default=None,
                   help="workspace to use (default: a new temp dir). Must not be BT Unet/ itself.")
    p.add_argument("--keep", action="store_true", help="keep the temp workspace afterwards")
    p.add_argument("--fail-fast", action="store_true")
    args = p.parse_args()

    os.environ.update(CPU_ENV)
    if sys.platform == "win32":
        # zmq (the notebook kernel channel) needs a selector loop on Windows.
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    workdir = (args.workdir or Path(tempfile.mkdtemp(prefix="bt_unet_smoke_"))).resolve()
    if workdir == REPO_DIR or workdir in REPO_DIR.parents:
        sys.exit("Refusing to use the repository folder as the workspace: it would "
                 "write into the real datasets/, saved_models/ and results/.")
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"Workspace: {workdir}\nPython:    {sys.executable}\n")

    nb_ws, script_ws = workdir / "nb_workspace", workdir / "script_workspace"
    executed, logs = workdir / "executed", workdir / "logs"

    stages = []
    if "notebooks" in args.only or "drivers" in args.only:
        kernel = install_kernel(workdir)
    if "notebooks" in args.only:
        make_workspace(nb_ws, args.datasets)
        stages += notebook_stages(nb_ws, args.datasets, kernel, executed)
    if {"scripts", "arch", "drivers"} & set(args.only):
        make_workspace(script_ws, args.datasets)
    if "scripts" in args.only:
        stages += script_stages(script_ws, args.datasets, logs)
    if "arch" in args.only:
        stages += arch_stages(script_ws, args.datasets, logs)
    if "drivers" in args.only:
        stages += driver_stages(script_ws, kernel, executed)

    results, t_all = [], time.time()
    for name, fn in stages:
        t0 = time.time()
        print(f"[ RUN  ] {name}", flush=True)
        try:
            fn()
            status, msg = "PASS", ""
        except Skip as e:
            status, msg = "SKIP", str(e)
        except Exception as e:  # noqa: BLE001 - report every failure, keep going
            status, msg = "FAIL", f"{type(e).__name__}: {e}"
            if not isinstance(e, (AssertionError, RuntimeError)):
                msg += "\n" + traceback.format_exc(limit=3)
        dt = time.time() - t0
        results.append((name, status, dt, msg))
        print(f"[ {status} ] {name}  ({dt:.0f}s)", flush=True)
        if msg:
            print("         " + msg.replace("\n", "\n         "), flush=True)
        if status == "FAIL" and args.fail_fast:
            break

    n_fail = sum(s == "FAIL" for _, s, _, _ in results)
    n_skip = sum(s == "SKIP" for _, s, _, _ in results)
    print(f"\n{'=' * 72}")
    for name, status, dt, msg in results:
        print(f"  {status}  {dt:6.0f}s  {name}" + (f"\n                  {msg}" if status == "SKIP" else ""))
    print(f"{'=' * 72}\n{len(results) - n_fail - n_skip}/{len(results)} passed, "
          f"{n_fail} failed, {n_skip} skipped in {time.time() - t_all:.0f}s")

    if n_fail or args.keep or args.workdir:
        print(f"Workspace kept at {workdir} (executed notebooks in executed/, script logs in logs/)")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
