"""Sequential frozen 72-job CPU queue; one progress line per verified job.

An explicit --job runs exactly one job in measurement/, never fits/.
No evaluation, scores, IC, network, GPU or scheduling.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import time
from tools import step5 as s
from tools import step5_runner as r


def run_all(bundle_dir, output, job=None):
    started = time.monotonic()
    bundle = r.require_admission(bundle_dir)
    names = [job] if job is not None else r.queue()
    base = s.OUTPUT_ROOT / ('measurement' if job else 'fits')
    output = Path(output).absolute()
    if not (output == base or output.is_relative_to(base)):
        raise ValueError('STOP queue output role')
    for i, name in enumerate(names, 1):
        before = time.monotonic()
        existed = (output / name).exists()
        r.run_job(bundle_dir, output, name, bundle)
        print(f'{i}/{len(names)} {name} {"RESUMED_VERIFIED" if existed else "COMPLETE_VERIFIED"} {time.monotonic()-before:.1f}s', flush=True)
    if job is not None:
        summary = dict(job=job, bundle_sha256=r.BUNDLE_SHA, admission_sha256=r.ADMISSION_SHA,
            wall_seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            fit_count=0 if existed else 1, ic_computed=False, label_statistics_computed=False,
            evidence_class='HISTORICAL_RESEARCH_CPU_ENGINEERING_ONLY')
        # Summary is outside the durable job directory; never invalidates its manifest.
        target = output / 'measurement.json'
        if not target.exists():
            s.write_json(target, summary)
    return len(names)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job', choices=r.queue())
    args = parser.parse_args(argv)
    run_all(args.bundle_dir, args.output, args.job)


if __name__ == '__main__':
    main()
