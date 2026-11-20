"""Render and submit the download (CPU) and run (GPU) jobs.

Never resubmits blindly: sbatch on SeaWulf can print "Socket timed out" and still create the job, so a
failed submit is followed by a look at the queue, not a retry.
"""
import os
import shlex
import string
import subprocess
import sys
from dataclasses import dataclass

from . import __version__
from .paths import repo

ACCOUNT = "pn_wosc091525s"
GPU_PARTITION = "a100"
CPU_PARTITION = "short-40core-shared"
CPUS_PER_GPU = 8
MEM_PER_GPU_GB = 60


def home():
    """Where runs, logs and the model cache live: $MODELRUNNER_HOME, else /gpfs/scratch/$USER/model-runner."""
    return os.environ.get("MODELRUNNER_HOME") or os.path.join("/gpfs/scratch", os.environ.get("USER", "user"),
                                                              "model-runner")


@dataclass
class Job:
    name: str
    script: str
    path: str
    log: str


def render(template, **values):
    with open(repo("slurm", template), encoding="utf-8") as f:
        return string.Template(f.read()).substitute(version=__version__, **values)


def run_job(model, out, run_args, time_limit="04:00:00", partition=GPU_PARTITION, cpu_test=False):
    """The GPU job; with cpu_test, a CPU-only job on a CPU partition (to test the path without a GPU)."""
    h = home()
    stem = os.path.splitext(os.path.basename(out))[0]
    name = f"mr-{stem}"[:40]
    log = os.path.join(h, "logs", f"{name}-%j.log")
    if cpu_test:
        partition, gres_line, cpus, mem = CPU_PARTITION, "# CPU-only test job (--cpu-test): no GPU", 8, "32G"
    else:
        gres_line = f"#SBATCH --gres=gpu:{model.gpus}"
        cpus, mem = CPUS_PER_GPU * model.gpus, f"{MEM_PER_GPU_GB * model.gpus}G"
    script = render("run.sbatch.tmpl", account=ACCOUNT, partition=partition, gres_line=gres_line,
                    cpus=cpus, mem=mem, time=time_limit,
                    job_name=name, log=log, hf_home=hf_home(), tmpdir=os.path.join(h, "tmp"),
                    python=sys.executable, run_args=" ".join(shlex.quote(a) for a in run_args), out=out,
                    results=out[:-5] + ".results.jsonl" if out.endswith(".json") else out + ".results.jsonl")
    return Job(name, script, os.path.join(h, "jobs", f"{name}.sbatch"), log)


def download_job(model):
    h = home()
    name = f"mr-dl-{model.hf_id.split('/')[-1]}"[:40]
    log = os.path.join(h, "logs", f"{name}-%j.log")
    script = render("download.sbatch.tmpl", account=ACCOUNT, partition=CPU_PARTITION, job_name=name, log=log,
                    hf_home=hf_home(), tmpdir=os.path.join(h, "tmp"), python=sys.executable,
                    hf_id=model.hf_id, revision=model.revision)
    return Job(name, script, os.path.join(h, "jobs", f"{name}.sbatch"), log)


def hf_home():
    return os.environ.get("HF_HOME") or os.path.join(home(), "hf")


def submit(job, after=None):
    """Write the script and sbatch it. Returns the job id. On an unclear failure, show the queue."""
    for d in (os.path.dirname(job.path), os.path.dirname(job.log), os.path.join(home(), "tmp")):
        os.makedirs(d, exist_ok=True)
    with open(job.path, "w", encoding="utf-8", newline="\n") as f:
        f.write(job.script)
    cmd = ["sbatch", "--parsable"] + ([f"--dependency=afterok:{after}"] if after else []) + [job.path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        queue = subprocess.run(["squeue", "--me", "-n", job.name], capture_output=True, text=True).stdout
        raise RuntimeError(f"sbatch failed: {p.stderr.strip()}\nIt may still have created the job; queue for "
                           f"{job.name}:\n{queue}\nCheck before submitting again.")
    return p.stdout.strip().split(";")[0]
