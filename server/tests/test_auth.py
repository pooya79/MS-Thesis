from server.app.core.auth import (
    SESSION_TTL_SECONDS,
    make_session_token,
    safe_next_path,
    verify_session_token,
)


def test_session_token_is_signed_and_expires() -> None:
    token = make_session_token("secret", now=1_000)

    assert verify_session_token(token, "secret", now=1_001)
    assert not verify_session_token(token, "wrong", now=1_001)
    assert not verify_session_token(token, "secret", now=1_000 + SESSION_TTL_SECONDS + 1)


def test_next_path_cannot_redirect_to_another_host() -> None:
    assert safe_next_path("/api/models?x=1") == "/api/models?x=1"
    assert safe_next_path("https://example.com") == "/"
    assert safe_next_path("//example.com") == "/"
