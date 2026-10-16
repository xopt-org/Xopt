"""Deterministic regression tests for per-session remote generator locks.

Run: python -m pytest xopt/tests/generators/remote/test_remote_session_locking.py -v --no-cov
"""

import base64
import threading
from concurrent.futures import ThreadPoolExecutor

import cloudpickle
import pytest

from xopt.generators.remote import server as upstream
from xopt.generators.remote.cloudpickle_server import get_cloudpickle_state


class _State:
    def __init__(self):
        self.value = 0


def _register(session_id, state):
    with upstream._lock:
        assert session_id not in upstream._sessions
        upstream._sessions[session_id] = upstream.GeneratorSession(generator=state)


def _remove(session_id):
    with upstream._lock:
        upstream._sessions.pop(session_id, None)


def test_same_session_snapshot_waits_for_mutation():
    """Export must not see a half-updated generator."""
    session_id = 'locking-same-session'
    state = _State()
    writer_entered = threading.Event()
    allow_writer_finish = threading.Event()
    reader_started = threading.Event()
    reader_done = threading.Event()
    _register(session_id, state)

    def writer():
        with upstream._locked_session(session_id) as obj:
            obj.value = 1
            writer_entered.set()
            assert allow_writer_finish.wait(timeout=10)
            obj.value = 2

    def reader():
        reader_started.set()
        response = get_cloudpickle_state(session_id)
        reader_done.set()
        return cloudpickle.loads(base64.b64decode(response['payload']))

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            write_future = pool.submit(writer)
            assert writer_entered.wait(timeout=10)
            read_future = pool.submit(reader)
            assert reader_started.wait(timeout=10)
            # While the writer owns the session lock, the reader must not complete.
            assert not reader_done.wait(timeout=0.2)
            allow_writer_finish.set()
            write_future.result(timeout=10)
            snapshot = read_future.result(timeout=10)
        assert snapshot.value == 2
        assert state.value == 2
    finally:
        allow_writer_finish.set()
        _remove(session_id)


def test_different_sessions_do_not_block_each_other():
    """A long-running operation in session A must not block session B."""
    first_id = 'locking-independent-a'
    second_id = 'locking-independent-b'
    entered_a = threading.Event()
    release_a = threading.Event()
    finished_b = threading.Event()
    _register(first_id, _State())
    _register(second_id, _State())

    def hold_a():
        with upstream._locked_session(first_id):
            entered_a.set()
            assert release_a.wait(timeout=10)

    def update_b():
        with upstream._locked_session(second_id) as obj:
            obj.value = 42
        finished_b.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(hold_a)
            assert entered_a.wait(timeout=10)
            future_b = pool.submit(update_b)
            assert finished_b.wait(timeout=5), 'Session B was blocked by session A'
            future_b.result(timeout=10)
            release_a.set()
            future_a.result(timeout=10)
        with upstream._locked_session(second_id) as obj:
            assert obj.value == 42
    finally:
        release_a.set()
        _remove(first_id)
        _remove(second_id)
