"""Collector health for the news-sentiment study: data-quality monitoring only.

Usage: ``python -m cryptosense.news.coverage [--data-dir PATH]``

Reads only ``data/news/runs.csv``. It never reads labels, sentiment or
prices, so it is allowed while the study runs (PREREGISTRATION_NEWS.md,
"allowed meanwhile: data-quality monitoring only").

It checks one coverage rule of section 6: a day needs a successful collector
run in at least ``MIN_COLLECTOR_HOURS`` hours. Hours are counted with
``aggregate.collector_hours``, the same function the study uses. The item
counts that H1 also needs are not checked here, so ``days_passing`` and
``max_possible_days`` are upper bounds on the days included for H1.
"""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptosense.news.aggregate import (
    FIRST_STUDY_DAY, LAST_STUDY_DAY, MIN_COLLECTOR_HOURS, collector_hours, complete_study_days, parse_utc,
)
from cryptosense.news.collect import DEFAULT_DATA_DIR
from cryptosense.news.label import _read_csv

# PREREGISTRATION_NEWS.md section 6: fewer qualifying days than this makes H1 inconclusive.
MIN_STUDY_DAYS_H1 = 60


def runs_until(runs, now):
    """Return the ``runs.csv`` rows of runs that started at or before ``now``."""
    return [row for row in runs if parse_utc(row["run_utc"]) <= now]


def last_successful_run(runs):
    """Return the start time (UTC datetime) of the latest run where every source was ok, or None."""
    statuses = {}
    for row in runs:
        statuses.setdefault(row["run_utc"], []).append(row["status"])
    ok = [parse_utc(run_utc) for run_utc, s in statuses.items() if all(status == "ok" for status in s)]
    return max(ok) if ok else None


def collector_status(runs, now):
    """Summarize collector coverage at time ``now`` from ``runs.csv`` rows.

    Runs that started after ``now`` are ignored, so the result depends only on
    what was known at ``now``.

    Parameters
    ----------
    runs : list of dict
        Rows of ``runs.csv``.
    now : datetime
        Timezone-aware current time.

    Returns
    -------
    dict
        ``days_complete``: study days that have ended; ``days_passing``: of
        those, days with at least ``MIN_COLLECTOR_HOURS`` hours; ``days_lost``:
        the rest; ``today``: today's date if it is a study day, else None;
        ``today_hours``: hours counted today so far; ``today_can_pass``: whether
        today can still reach ``MIN_COLLECTOR_HOURS``; ``days_ahead``: study
        days after today; ``max_possible_days``: the most days that can still
        pass; ``days_to_spare``: ``max_possible_days - MIN_STUDY_DAYS_H1``
        (negative means H1 cannot reach the minimum); ``last_ok_run``: the
        latest fully successful run (datetime or None); ``lost_days``: list of
        ``(date, hours)`` for the lost days.
    """
    now = now.astimezone(timezone.utc)
    known = runs_until(runs, now)
    hours = collector_hours(known)
    complete = complete_study_days(now)
    passing = [day for day in complete if hours.get(day, 0) >= MIN_COLLECTOR_HOURS]
    lost = [(day, hours.get(day, 0)) for day in complete if hours.get(day, 0) < MIN_COLLECTOR_HOURS]

    day = now.date()
    in_study = FIRST_STUDY_DAY <= day <= LAST_STUDY_DAY
    today_hours = hours.get(day, 0) if in_study else 0
    if in_study:
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        current_counted = collector_hours(r for r in known if parse_utc(r["run_utc"]) >= hour_start).get(day, 0)
        hours_left = 24 - now.hour - current_counted  # the current hour can still count if not counted yet
        today_can_pass = today_hours + hours_left >= MIN_COLLECTOR_HOURS
    else:
        today_can_pass = False
    first_ahead = max(day + timedelta(days=1), FIRST_STUDY_DAY)
    days_ahead = max((LAST_STUDY_DAY - first_ahead).days + 1, 0)
    max_possible = len(passing) + int(today_can_pass) + days_ahead
    return {
        "days_complete": len(complete), "days_passing": len(passing), "days_lost": len(lost),
        "today": day if in_study else None, "today_hours": today_hours, "today_can_pass": today_can_pass,
        "days_ahead": days_ahead, "max_possible_days": max_possible,
        "days_to_spare": max_possible - MIN_STUDY_DAYS_H1, "last_ok_run": last_successful_run(known),
        "lost_days": lost,
    }


def format_status(status, now):
    """Format a ``collector_status`` result as plain-language lines (no sentiment, no item counts)."""
    now = now.astimezone(timezone.utc)
    last = status["last_ok_run"]
    lines = []
    if last is None:
        lines.append("last successful run: none")
    else:
        age = int((now - last).total_seconds() // 3600)
        lines.append(f"last successful run: {last:%Y-%m-%d %H:%M} UTC ({age} h ago)")
    if status["today"] is not None:
        verdict = "can still count" if status["today_can_pass"] else "cannot count any more"
        lines.append(f"today ({status['today']}): {status['today_hours']} of {MIN_COLLECTOR_HOURS} hours, {verdict}")
    lines += [
        f"study days ended: {status['days_complete']}, passing the hours rule: {status['days_passing']}, "
        f"lost: {status['days_lost']}",
        f"most days still possible: {status['max_possible_days']} (H1 needs {MIN_STUDY_DAYS_H1}; "
        f"days to spare: {status['days_to_spare']})",
    ]
    lines += [f"  lost {day}: {hours} hours" for day, hours in status["lost_days"]]
    return lines


def load_collector_status(data_dir, now):
    """Read ``data_dir / "news" / "runs.csv"`` and return ``collector_status`` at ``now``.

    A missing file counts as no runs.
    """
    return collector_status(_read_csv(Path(data_dir) / "news" / "runs.csv"), now)


def main(argv=None, now=None):
    """Command-line entry point: print the collector status.

    ``now`` is ``now() -> datetime``; inject a fixed clock in tests.
    """
    parser = argparse.ArgumentParser(description="Collector health for the news-sentiment study.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="project data directory (default: %(default)s)")
    args = parser.parse_args(argv)
    moment = (now or (lambda: datetime.now(timezone.utc)))()
    status = load_collector_status(args.data_dir, moment)
    print("\n".join(format_status(status, moment)))


if __name__ == "__main__":
    main()
