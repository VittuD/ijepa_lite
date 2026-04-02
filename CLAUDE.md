# CLAUDE.md — Project conventions for Claude Code

## Testing

**Do NOT run tests or import-checks locally.** The project runs on a remote HPC cluster (SLURM/sbatch). All validation is done by submitting a debug sbatch job on the cluster.

Never run `python -c "import ..."`, `pytest`, or any training/eval script locally to verify changes.
