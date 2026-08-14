"""The HTTP and websocket surface, against a scripted recogniser."""

from __future__ import annotations

import pytest


# ======================================================================
# System
# ======================================================================


def test_health(client):
    test_client, _ = client
    body = test_client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["classes"] == 26


def test_config_exposes_the_tunables_but_no_secrets(client):
    test_client, _ = client
    body = test_client.get("/api/config").json()
    assert len(body["classes"]) == 26
    assert body["stable_frames"] == 3
    assert body["gemini_enabled"] is False
    assert body["gemini_model"] is None
    assert "api_key" not in str(body).lower()


def test_openapi_document_builds(client):
    test_client, _ = client
    assert test_client.get("/openapi.json").status_code == 200


# ======================================================================
# Sessions and text
# ======================================================================


def test_session_lifecycle(client):
    test_client, _ = client
    created = test_client.post("/api/session")
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    for character in "HELLO":
        test_client.post(
            "/api/session/text",
            json={"session_id": session_id, "command": "add_character", "value": character},
        )
    body = test_client.get(f"/api/session/{session_id}").json()
    assert body["text"]["text"] == "HELLO"

    reset = test_client.post(
        "/api/session/reset", json={"session_id": session_id, "command": "clear"}
    )
    assert reset.json()["text"]["text"] == ""


def test_unknown_session_is_404(client):
    test_client, _ = client
    assert test_client.get("/api/session/does-not-exist").status_code == 404


def test_invalid_command_is_rejected(client):
    test_client, _ = client
    response = test_client.post(
        "/api/session/text", json={"command": "explode", "value": ""}
    )
    assert response.status_code == 422


def test_command_missing_its_value_is_rejected(client):
    test_client, _ = client
    response = test_client.post(
        "/api/session/text", json={"command": "add_character", "value": ""}
    )
    assert response.status_code == 422


def test_oversized_text_is_rejected(client):
    test_client, _ = client
    response = test_client.post(
        "/api/session/text",
        json={"command": "set_text", "value": "x" * 5000},
    )
    assert response.status_code == 422


# ======================================================================
# Suggestions
# ======================================================================


def test_suggestions_for_an_empty_buffer(client):
    test_client, _ = client
    body = test_client.post(
        "/api/suggestions?wait_for_llm=false",
        json={"text": "", "current_word": "", "context": []},
    ).json()
    assert body["source"] == "local"
    assert body["word_suggestions"]


def test_suggestions_complete_a_partial_word(client):
    test_client, _ = client
    body = test_client.post(
        "/api/suggestions?wait_for_llm=false",
        json={"text": "HEL", "current_word": "HEL", "context": []},
    ).json()
    assert all(word.startswith("HEL") for word in body["word_suggestions"])


def test_suggestions_use_multi_word_context(client):
    test_client, _ = client
    body = test_client.post(
        "/api/suggestions?wait_for_llm=false",
        json={"text": "HOW ARE", "current_word": "", "context": ["HOW", "ARE"]},
    ).json()
    assert "YOU" in body["word_suggestions"]


def test_suggestions_work_without_gemini(client):
    """The key resilience property: no API key must not mean no suggestions."""
    test_client, _ = client
    body = test_client.post(
        "/api/suggestions",
        json={"text": "HELLO", "current_word": "", "context": ["HELLO"]},
    ).json()
    assert body["word_suggestions"]
    assert body["llm_pending"] is False


def test_suggestion_request_is_validated(client):
    test_client, _ = client
    response = test_client.post(
        "/api/suggestions", json={"text": "hi", "max_words": 99}
    )
    assert response.status_code == 422


# ======================================================================
# Signs
# ======================================================================


def test_sign_catalog(client):
    test_client, _ = client
    body = test_client.get("/api/signs").json()
    assert len(body["signs"]) == 26
    assert body["signs"]["A"] == "/signs/A.svg"


def test_sign_assets_are_served(client):
    test_client, _ = client
    response = test_client.get("/signs/A.svg")
    assert response.status_code == 200
    assert "svg" in response.headers["content-type"]


def test_translate_to_sign(client):
    test_client, _ = client
    body = test_client.post("/api/translate-to-sign", json={"text": "HI"}).json()
    assert body["sign_count"] == 2
    assert body["words"][0]["signs"][0]["asset"] == "/signs/H.svg"


def test_translate_reports_characters_with_no_sign(client):
    test_client, _ = client
    body = test_client.post("/api/translate-to-sign", json={"text": "Café"}).json()
    assert body["unsupported"] == ["É"]


# ======================================================================
# Recognition
# ======================================================================


def test_predict_with_no_hand(client, blank_frame):
    test_client, recognizer = client
    recognizer.script = [(None, 0.0)]
    body = test_client.post("/api/predict", json={"image": blank_frame}).json()
    assert body["hand_detected"] is False
    assert body["status"] == "idle"
    assert body["committed_letter"] is None


def test_predict_commits_after_enough_stable_frames(client, blank_frame):
    test_client, recognizer = client
    recognizer.script = [("A", 0.97)]
    session_id = test_client.post("/api/session").json()["session_id"]

    committed = [
        test_client.post(
            "/api/predict", json={"image": blank_frame, "session_id": session_id}
        ).json()["committed_letter"]
        for _ in range(6)
    ]
    assert committed.count("A") == 1          # stable_frames is 3 in tests
    assert test_client.get(f"/api/session/{session_id}").json()["text"]["text"] == "A"


def test_predict_ignores_low_confidence(client, blank_frame):
    test_client, recognizer = client
    recognizer.script = [("A", 0.30)]
    session_id = test_client.post("/api/session").json()["session_id"]
    for _ in range(8):
        body = test_client.post(
            "/api/predict", json={"image": blank_frame, "session_id": session_id}
        ).json()
    assert body["status"] == "low_confidence"
    assert test_client.get(f"/api/session/{session_id}").json()["text"]["text"] == ""


def test_predict_can_report_without_touching_the_buffer(client, blank_frame):
    test_client, recognizer = client
    recognizer.script = [("B", 0.99)]
    session_id = test_client.post("/api/session").json()["session_id"]
    for _ in range(6):
        test_client.post(
            "/api/predict",
            json={
                "image": blank_frame,
                "session_id": session_id,
                "apply_to_buffer": False,
            },
        )
    assert test_client.get(f"/api/session/{session_id}").json()["text"]["text"] == ""


@pytest.mark.parametrize("payload", ["not-base64!!", "", "aGVsbG8="])
def test_predict_rejects_junk_images(client, payload):
    test_client, _ = client
    response = test_client.post("/api/predict", json={"image": payload})
    assert response.status_code == 422


# ======================================================================
# WebSocket
# ======================================================================


def test_socket_greets_with_ready_and_suggestions(client):
    test_client, _ = client
    with test_client.websocket_connect("/ws/recognition") as socket:
        ready = socket.receive_json()
        assert ready["type"] == "ready"
        assert len(ready["classes"]) == 26
        assert socket.receive_json()["type"] == "suggestions"


def test_socket_streams_recognition_and_commits_a_letter(client, blank_frame):
    test_client, recognizer = client
    recognizer.script = [("C", 0.99)]

    with test_client.websocket_connect("/ws/recognition") as socket:
        socket.receive_json()   # ready
        socket.receive_json()   # opening suggestions

        committed = None
        for _ in range(4):
            socket.send_json({"type": "frame", "image": blank_frame})
            message = socket.receive_json()
            assert message["type"] == "recognition"
            assert message["hand_detected"] is True
            if message["committed_letter"]:
                committed = message["committed_letter"]
                assert message["text"]["text"] == "C"
                assert socket.receive_json()["type"] == "suggestions"
                break
        assert committed == "C"


def test_socket_editing_commands_update_the_buffer(client):
    test_client, _ = client
    with test_client.websocket_connect("/ws/recognition") as socket:
        socket.receive_json()
        socket.receive_json()

        for character in "HI":
            socket.send_json(
                {"type": "command", "command": "add_character", "value": character}
            )
            assert socket.receive_json()["type"] == "text"
            assert socket.receive_json()["type"] == "suggestions"

        socket.send_json({"type": "command", "command": "backspace"})
        assert socket.receive_json()["text"]["text"] == "H"


def test_socket_reports_a_bad_frame_without_closing(client):
    test_client, _ = client
    with test_client.websocket_connect("/ws/recognition") as socket:
        socket.receive_json()
        socket.receive_json()

        socket.send_json({"type": "frame", "image": "!!!not base64!!!"})
        error = socket.receive_json()
        assert error["type"] == "error"
        assert error["fatal"] is False

        socket.send_json({"type": "ping"})
        assert socket.receive_json()["type"] == "pong"


def test_socket_rejects_unknown_messages_without_closing(client):
    test_client, _ = client
    with test_client.websocket_connect("/ws/recognition") as socket:
        socket.receive_json()
        socket.receive_json()

        socket.send_text("this is not json")
        assert socket.receive_json()["code"] == "bad_json"

        socket.send_json({"type": "command", "command": "detonate"})
        assert socket.receive_json()["code"] == "unknown_command"

        socket.send_json({"type": "ping"})
        assert socket.receive_json()["type"] == "pong"


def test_socket_types_a_word_from_a_realistic_noisy_stream(client, blank_frame):
    """Regression: 'the letter is detected but the bar never fills'.

    Confidences here are what a live webcam actually produces (0.5-0.9, with
    the odd wrong letter), not the near-1.0 the model scores on its own
    training data. The old consecutive-frame stabilizer typed nothing at all
    from a stream like this.
    """
    test_client, recognizer = client
    recognizer.script = [
        # reaching for the sign
        (None, 0.0), (None, 0.0),
        # holding H, with one misread frame and ordinary wobble
        ("H", 0.48), ("H", 0.71), ("N", 0.52), ("H", 0.83),
        ("H", 0.77), ("H", 0.90), ("H", 0.86), ("H", 0.91),
    ]

    with test_client.websocket_connect("/ws/recognition") as socket:
        socket.receive_json()   # ready
        socket.receive_json()   # opening suggestions

        progresses: list[float] = []
        typed: list[str] = []
        for _ in range(len(recognizer.script)):
            socket.send_json({"type": "frame", "image": blank_frame})
            message = socket.receive_json()
            assert message["type"] == "recognition"
            progresses.append(message["progress"])
            if message["committed_letter"]:
                typed.append(message["committed_letter"])
                socket.receive_json()   # the suggestions that follow

        assert typed == ["H"], f"nothing typed; progress was {progresses}"
        # The bar must have visibly moved before committing, not jumped 0 -> 1.
        moving = [p for p in progresses if 0.0 < p < 1.0]
        assert len(moving) >= 3, f"bar barely moved: {progresses}"


def test_socket_resumes_an_existing_session(client):
    test_client, _ = client
    session_id = test_client.post("/api/session").json()["session_id"]
    test_client.post(
        "/api/session/text",
        json={"session_id": session_id, "command": "set_text", "value": "HELLO"},
    )
    with test_client.websocket_connect(
        f"/ws/recognition?session_id={session_id}"
    ) as socket:
        assert socket.receive_json()["session_id"] == session_id
        assert socket.receive_json()["text"]["text"] == "HELLO"
