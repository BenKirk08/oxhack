#tests for presage_check.py + the Presage card at the bottom of the video check tab
#no node, internet or presage credits needed -> made up readings + a fake measure.mjs answer
#importing app.py loads gradio so this file takes a few secs
import json
import subprocess
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import app
import presage_check
from presage_check import PresageReading, presage_bpm, run_presage


def reading(
    bpms: list[float], stable: list[bool], hints: dict[str, int] | None = None
) -> PresageReading:
    #a made up presage result, 1 reading per second
    count = len(bpms)
    return PresageReading(
        times=np.arange(count, dtype=float),
        bpms=np.array(bpms, dtype=float),
        confidences=np.full(count, 50.0),  #confidence doesnt matter here, stable does
        stable=np.array(stable, dtype=bool),
        hints=hints or {},  #or -> {} if None was given
    )


def test_bpm_is_the_median_of_the_stable_readings() -> None:
    assert presage_bpm(reading([70, 72, 90], [True, True, False])) == (71.0, True)  #90 ignored


def test_bpm_without_stable_readings_is_flagged_unsure() -> None:
    assert presage_bpm(reading([80, 84], [False, False])) == (82.0, False)


def test_no_readings_means_no_bpm() -> None:
    assert presage_bpm(reading([], [])) == (None, False)


def test_card_shows_the_number() -> None:
    page = app.presage_html(reading([70, 72], [True, True]))
    assert "71<small>BPM</small>" in page
    assert "low confidence" not in page


def test_card_says_when_presage_is_unsure() -> None:
    assert "low confidence" in app.presage_html(reading([80], [False]))


def test_card_without_readings_shows_presages_warning() -> None:
    hints = {"Too dark.": 3, "Place more of the chest in view.": 40}
    page = app.presage_html(reading([], [], hints))
    assert "No reading" in page
    assert "Place more of the chest in view." in page  #the most common one


def test_card_escapes_the_problem_text() -> None:
    assert "&lt;b&gt;bad&lt;/b&gt;" in app.presage_html(problem="<b>bad</b>")


def test_card_before_any_video() -> None:
    assert "Analyse a video to compare" in app.presage_html()


@pytest.fixture
def fake_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # node "installed" + SDK folder and .env in a temp folder -> run_presage gets past its checks
    (tmp_path / "node_modules").mkdir()
    (tmp_path / ".env").write_text("SMARTSPECTRA_API_KEY=test\n")
    monkeypatch.setattr(presage_check.shutil, "which", lambda name: "/usr/bin/node")
    monkeypatch.setattr(presage_check, "MEASURE_SCRIPT", tmp_path / "measure.mjs")
    monkeypatch.setattr(presage_check, "ENV_PATH", tmp_path / ".env")


def fake_measure(monkeypatch: pytest.MonkeyPatch, output: dict[str, Any]) -> None:
    #swaps running measure.mjs for a made up answer (monkeypatch undoes it after the test)
    finished = subprocess.CompletedProcess([], 0, stdout=json.dumps(output), stderr="")
    monkeypatch.setattr(presage_check.subprocess, "run", lambda *args, **kwargs: finished)


def test_run_presage_reads_measures_json(fake_setup: None, monkeypatch: pytest.MonkeyPatch) -> None:
    pulse = [[12.0, 71.0, 55.0, 1], [13.0, 73.0, 30.0, 0]]  #[time, BPM, confidence, stable]
    fake_measure(monkeypatch, {"pulse": pulse, "breathing": [], "hints": {"Too dark.": 4},
                               "errors": []})
    found = run_presage(Path("clip.mov"))
    assert list(found.bpms) == [71.0, 73.0]
    assert list(found.stable) == [True, False]
    assert found.hints == {"Too dark.": 4}


def test_run_presage_passes_on_presages_error(
    fake_setup: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = {"code": 5, "message": "A network error occurred."}
    fake_measure(monkeypatch, {"pulse": [], "breathing": [], "hints": {}, "errors": [error]})
    with pytest.raises(RuntimeError, match="network error"):
        run_presage(Path("clip.mov"))


@pytest.mark.parametrize(
    ("problem", "shown"),
    [(RuntimeError("Presage: no internet"), "no internet"), (ValueError("bug"), "in the terminal")],
)
def test_presage_failing_only_changes_its_card(
    problem: Exception, shown: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    #any presage error ends up as text on its card, never a crash that stops the page
    def broken(path: Path) -> PresageReading:
        raise problem

    monkeypatch.setattr(app, "run_presage", broken)
    job: Future[PresageReading | str] = Future()
    app.presage_job(job, Path("clip.mov"))  #runs right here, not on a thread
    monkeypatch.setattr(app, "PRESAGE_JOBS", {"clip.mov": job})
    assert shown in app.presage_status("clip.mov")


def test_nothing_uploaded_leaves_the_card_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "PRESAGE_JOBS", {})
    with pytest.raises(app.gr.Error):  #analyse_upload stops b4 starting presage
        app.analyse_upload(None, "POS")
    assert app.PRESAGE_JOBS == {}  #so no paid presage run
    assert "Analyse a video to compare" in app.presage_status(None)


def test_card_says_measuring_until_presage_is_done(monkeypatch: pytest.MonkeyPatch) -> None:
    job: Future[PresageReading | str] = Future()
    monkeypatch.setattr(app, "PRESAGE_JOBS", {"a.mov": job})
    assert "Measuring with Presage" in app.presage_status("a.mov")
    job.set_result(reading([70, 72], [True, True]))
    assert "71<small>BPM</small>" in app.presage_status("a.mov")
    assert "Analyse a video to compare" in app.presage_status("b.mov")  #a's never shows on b


def test_presage_runs_once_per_video_but_retries_a_failure(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = []
    answers = iter([RuntimeError("Presage: no internet"), reading([70], [True])])

    def fake_run(path: Path) -> PresageReading:
        runs.append(path)
        answer = next(answers)  #1st run fails, 2nd works
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(app, "run_presage", fake_run)
    monkeypatch.setattr(app, "PRESAGE_JOBS", {})
    app.start_presage("a.mov")
    app.PRESAGE_JOBS["a.mov"].result(timeout=5)  #waits for its thread
    app.start_presage("a.mov")  #last try failed -> runs again
    app.PRESAGE_JOBS["a.mov"].result(timeout=5)
    app.start_presage("a.mov")  #worked this time -> not run a 3rd time
    assert len(runs) == 2
    assert "70<small>BPM</small>" in app.presage_status("a.mov")
