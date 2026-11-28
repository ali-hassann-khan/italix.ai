import sys

import httpx
import pytest

from scripts import phase11_training as launcher


def test_launcher_requires_matching_confirmation_and_review(monkeypatch):
    monkeypatch.setenv("ITALIX_OPERATOR_TOKEN", "fixture-token-for-an-authenticated-local-server")
    calls = []
    verified = []
    plan = {"id": launcher.JOB_ID, "candidate_id": launcher.CANDIDATE, "review_digest": launcher.REVIEW_DIGEST}

    def verify():
        verified.append(True)
        return plan

    send = httpx.Client

    def response(request):
        calls.append((request.method, request.url.path))
        assert request.headers["authorization"].startswith("Bearer ")
        if request.method == "GET":
            return httpx.Response(200, json=plan)
        assert request.url.path.endswith("/execute")
        assert launcher.REVIEW_DIGEST.encode() in request.content
        return httpx.Response(202, json={"id": launcher.JOB_ID, "status": "running"})

    monkeypatch.setattr(launcher, "verify", verify)
    monkeypatch.setattr(launcher.httpx, "Client", lambda **kwargs: send(
        transport=httpx.MockTransport(response), **kwargs,
    ))
    monkeypatch.setattr(sys, "argv", ["launcher", "start", "--confirm", "wrong"])
    with pytest.raises(SystemExit):
        launcher.main()
    assert not verified and not calls

    monkeypatch.setattr(sys, "argv", ["launcher", "start", "--confirm", launcher.CANDIDATE])
    launcher.main()
    assert verified == [True]
    assert calls == [("GET", "/v1/training/jobs/1/review"), ("POST", "/v1/training/jobs/1/execute")]


@pytest.mark.parametrize("terminal", ["completed", "failed"])
def test_watch_exits_at_terminal_state(monkeypatch, capsys, terminal):
    responses = iter([{"status": "running"}, {"status": terminal}])
    delays = []
    monkeypatch.setattr(launcher, "read_job", lambda client: next(responses))
    monkeypatch.setattr(launcher.time, "sleep", delays.append)
    launcher.watch(None)
    assert delays == [3]
    assert f'"status": "{terminal}"' in capsys.readouterr().out
