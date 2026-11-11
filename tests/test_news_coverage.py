import ast
import csv
from datetime import date, datetime, timezone
from pathlib import Path

from cryptosense.news import coverage as module
from cryptosense.news.aggregate import LAST_STUDY_DAY, MIN_COLLECTOR_HOURS
from cryptosense.news.collect import RUN_COLUMNS
from cryptosense.news.coverage import (
    MIN_STUDY_DAYS_H1, collector_status, format_status, last_successful_run, load_collector_status, main,
    runs_until,
)
from cryptosense.news.feeds import FEEDS

SOURCES = [feed["source"] for feed in FEEDS]
TOTAL_STUDY_DAYS = 99  # 2026-10-25 to 2027-01-31


def utc(text):
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def runs_for(day, hours=range(24), failed=(), minute=5):
    """Collector runs at ``minute`` of each given hour of ``day``; hours in ``failed`` have one failed source."""
    rows = []
    for hour in hours:
        run_utc = f"{day}T{hour:02d}:{minute:02d}:00Z"
        for position, source in enumerate(SOURCES):
            status = "failed" if hour in failed and position == 0 else "ok"
            rows.append({"run_utc": run_utc, "source": source, "status": status, "new": "0",
                         "duplicate": "0", "skipped": "0", "error": ""})
    return rows


def test_full_and_short_days_are_counted():
    runs = runs_for("2026-10-25") + runs_for("2026-10-26", hours=range(17))
    status = collector_status(runs, utc("2026-10-27T00:30:00Z"))
    assert status["days_complete"] == 2
    assert status["days_passing"] == 1
    assert status["days_lost"] == 1
    assert status["lost_days"] == [(date(2026, 10, 26), 17)]


def test_exactly_min_hours_passes_and_failed_runs_do_not_count():
    runs = runs_for("2026-10-25", hours=range(MIN_COLLECTOR_HOURS))
    runs += runs_for("2026-10-26", hours=range(MIN_COLLECTOR_HOURS), failed={0})
    status = collector_status(runs, utc("2026-10-27T00:30:00Z"))
    assert status["days_passing"] == 1
    assert status["lost_days"] == [(date(2026, 10, 26), MIN_COLLECTOR_HOURS - 1)]


def test_max_possible_and_spare_days():
    runs = runs_for("2026-10-25")  # 26 Oct gets nothing
    status = collector_status(runs, utc("2026-10-27T00:30:00Z"))
    # 27 Oct (today) can still pass, then 28 Oct .. 31 Jan
    assert status["today"] == date(2026, 10, 27)
    assert status["today_can_pass"] is True
    assert status["days_ahead"] == TOTAL_STUDY_DAYS - 3
    assert status["max_possible_days"] == 1 + 1 + TOTAL_STUDY_DAYS - 3
    assert status["days_to_spare"] == status["max_possible_days"] - MIN_STUDY_DAYS_H1


def test_today_can_pass_depends_on_hours_left():
    # 7 hours counted by 17:30, current hour already counted: 7 + 6 hours left = 13 < 18
    runs = runs_for("2026-11-11", hours=range(11, 18))
    status = collector_status(runs, utc("2026-11-11T17:30:00Z"))
    assert status["today_hours"] == 7
    assert status["today_can_pass"] is False
    # 12 hours by 17:30, current hour already counted: 12 + 6 = 18
    runs = runs_for("2026-11-11", hours=range(6, 18))
    assert collector_status(runs, utc("2026-11-11T17:30:00Z"))["today_can_pass"] is True
    # 11 hours by 17:00 with the 17:00 hour not run yet: 11 + 7 = 18
    runs = runs_for("2026-11-11", hours=range(6, 17))
    assert collector_status(runs, utc("2026-11-11T17:00:00Z"))["today_can_pass"] is True


def test_truncation_leakage_runs_after_now_are_ignored():
    now = utc("2026-11-05T12:30:00Z")
    past = runs_for("2026-11-04") + runs_for("2026-11-05", hours=range(12))
    # a run later in the current hour, the rest of today, and tomorrow
    future = (runs_for("2026-11-05", hours=[12], minute=40) + runs_for("2026-11-05", hours=range(13, 24))
              + runs_for("2026-11-06"))
    assert collector_status(past + future, now) == collector_status(past, now)
    assert runs_until(past + future, now) == past


def test_run_exactly_at_now_is_known():
    runs = runs_for("2026-11-05", hours=[12], minute=30)
    assert collector_status(runs, utc("2026-11-05T12:30:00Z"))["today_hours"] == 1


def test_last_successful_run_skips_failed_runs():
    runs = runs_for("2026-11-05", hours=[8, 9], failed={9})
    assert last_successful_run(runs) == utc("2026-11-05T08:05:00Z")
    assert last_successful_run(runs_for("2026-11-05", hours=[9], failed={9})) is None
    assert last_successful_run([]) is None


def test_after_study_end():
    status = collector_status(runs_for("2027-01-31"), utc("2027-02-02T10:00:00Z"))
    assert status["today"] is None
    assert status["days_ahead"] == 0
    assert status["days_complete"] == TOTAL_STUDY_DAYS
    assert status["max_possible_days"] == status["days_passing"] == 1
    assert LAST_STUDY_DAY == date(2027, 1, 31)


def test_before_study_start():
    status = collector_status([], utc("2026-10-24T23:00:00Z"))
    assert status["days_complete"] == 0
    assert status["today"] is None
    assert status["max_possible_days"] == TOTAL_STUDY_DAYS


def test_format_status_lines():
    now = utc("2026-10-27T00:30:00Z")
    lines = format_status(collector_status(runs_for("2026-10-25"), now), now)
    assert lines[0] == "last successful run: 2026-10-25 23:05 UTC (25 h ago)"
    assert lines[1] == "today (2026-10-27): 0 of 18 hours, can still count"
    assert "lost 2026-10-26: 0 hours" in lines[-1]


def write_runs(tmp_path, rows):
    path = tmp_path / "news" / "runs.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_main_reads_runs_file(tmp_path, capsys):
    write_runs(tmp_path, runs_for("2026-10-25"))
    main(["--data-dir", str(tmp_path)], now=lambda: utc("2026-10-26T01:00:00Z"))
    out = capsys.readouterr().out
    assert "passing the hours rule: 1, lost: 0" in out


def test_load_collector_status_matches_collector_status(tmp_path):
    rows = runs_for("2026-10-25")
    write_runs(tmp_path, rows)
    now = utc("2026-10-26T01:00:00Z")
    assert load_collector_status(tmp_path, now) == collector_status(rows, now)


def test_main_without_runs_file(tmp_path, capsys):
    main(["--data-dir", str(tmp_path)], now=lambda: utc("2026-10-26T01:00:00Z"))
    assert "last successful run: none" in capsys.readouterr().out


def test_module_never_touches_labels_or_prices():
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "labels.csv" not in source and "articles.csv" not in source
    imported = {node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)}
    assert "cryptosense.data" not in imported
