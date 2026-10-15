#!/usr/bin/env python3
"""Serialize Numi resting-scene preparation, exact audits and native runs.

Usage: python tools/run_resting_job_exclusive.py -- COMMAND [ARG ...]
Use the same lock on one host for every substantial geometry/solver job.
The standard-library advisory lock survives exec and releases on process exit,
including an interrupted or orphaned job. It does not change solver settings,
assets, accepted state, output formats, or the child command's exit status.
Do not unlink a live lock file; inspect holders with lsof if needed.
"""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import stat
import sys
import tempfile

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock",type=Path,
        default=Path(tempfile.gettempdir())/f"numi-resting-heavy-{os.getuid()}.lock")
    parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    command=args.command
    if command[:1]==["--"]:
        command=command[1:]
    if not command:
        parser.error("a child command is required after --")
    fd=os.open(args.lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    info=os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1:
        os.close(fd)
        parser.error("lock must be a regular, singly linked file owned by the current user")
    print(f"resting_job_lock waiting pid={os.getpid()}",file=sys.stderr,flush=True)
    fcntl.flock(fd,fcntl.LOCK_EX)
    os.set_inheritable(fd,True)
    print(f"resting_job_lock acquired pid={os.getpid()}",file=sys.stderr,flush=True)
    try:
        os.execvp(command[0],command)
    except OSError:
        os.close(fd)
        raise

if __name__=="__main__":
    raise SystemExit(main())
