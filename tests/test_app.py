#tests for the Video Check tabs verdict box (step 14)
#every verdict turns into html ok + shows its headline
# text gets escaped so a "<" in a reason cant break the page
#importing app.py loads gradio so this file takes a few secs
from dataclasses import replace
from typing import Any

import pytest
from test_detector import make_tracks

import app
from detector import judge, video_features


@pytest.mark.parametrize(
    ("options", "headline"), #each run = one made up video + the headline it must show
    [
        ({"neck_pulse": 0.004}, "Likely real"),
        ({"inner_pulse": 0.0, "outer_pulse": 0.0, "neck_pulse": 0.004}, "Likely AI-generated"),
        ({"pulse": 0.0}, "Not enough signal to judge"),
    ],
)
def test_verdict_box_shows_every_kind_of_verdict(options: dict[str, Any], headline: str) -> None:
    features = video_features(make_tracks(**options), 0.9)
    page = app.verdict_html(judge(features), features)
    assert headline in page
    assert "How it decided" in page
    assert "✓" in page or "✗" in page  #checks list is there


def test_verdict_box_escapes_its_text() -> None:
    features = video_features(make_tracks(), 0.9)
    verdict = replace(judge(features), reasons=["<b>sneaky</b>"])
    assert "&lt;b&gt;sneaky&lt;/b&gt;" in app.verdict_html(verdict, features)


def test_waiting_box_before_any_video() -> None:
    assert "Upload a video, then Analyse" in app.verdict_html(None)
