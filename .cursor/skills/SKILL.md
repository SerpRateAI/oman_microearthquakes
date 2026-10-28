---
name: oman-microearthquakes
description: Procedures for creating and running Python scripts in the Oman microearthquakes project. Use when adding a new analysis script, writing or editing a Slurm submission script in ~/slurm_oman, deciding whether a script is safe to run interactively, or preparing to run or submit a job for seismic analysis, station processing, body wave attenuation, or microearthquake catalogs.
---

# Oman microearthquakes project procedures

Persistent style conventions for this project (import style, argument handling,
conda environment) live in `.cursor/rules/oman-python.mdc` and apply
regardless of this skill. This file covers the procedures for creating,
submitting, and running scripts.

## Adding a new analysis script

1. Write the Python script following the project rule conventions.
2. Create a matching Slurm submission script (see below). Every new analysis
   script gets one, even if it will first be run interactively.
3. Verify statically — do not test-run the full script. See "Deciding whether
   a script is safe to run" below.

When modifying existing project scripts, preserve the project's existing
naming, file organization, and scientific assumptions unless the user
explicitly asks for a refactor.

## Slurm submission scripts

Submission scripts live in `~/slurm_oman` and are named `run_$(script_name).sh`.

Before writing a new one, read an existing submission script in that directory
and match its structure, resource directives, and output-path conventions.
Do not invent Slurm directives, partition names, or account strings — copy them
from an existing script and adjust only what the new job requires.

`~/slurm_oman` is outside the repository. Expand `~` explicitly rather than
assuming the shell will.

## Deciding whether a script is safe to run

Treat a script as long-running if any of the following are true:

- it reads continuous waveform data,
- it loops over multiple stations, days, or events,
- it performs spectral, CWT, or inversion computations over a full dataset,
- you cannot confirm from the code that it completes in under a minute.

When in doubt, assume the script is long-running.

## Handling long-running scripts

- Create or edit the Slurm submission script.
- Do not execute the full Python command.
- Do not run the job with `sbatch` unless the user explicitly asks.
- Verify with static or lightweight checks only:
  - checking imports and argument parsing,
  - reviewing file paths,
  - checking Slurm directives,
  - running `conda run -n oman_new python -m py_compile script_name.py`.

Ask before performing any command that could start a long computation, submit a
job, overwrite large outputs, or consume substantial cluster resources.

## Before finishing

State briefly:

- whether a `run_$(script_name).sh` exists in `~/slurm_oman` for any new script,
- how the script was classified (long-running or not) and why,
- which verification steps were run, and which were skipped as too expensive.
