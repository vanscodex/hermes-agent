"""``providers.bedrock.request_timeout_seconds`` must reach the boto3 client (#133318).

botocore's default ``read_timeout=60`` applies to ``converse_stream`` when no
``botocore.config.Config`` is passed, so a reasoning model still prefilling or
thinking at 60s is killed by the socket well before ``_BedrockStream``'s own
derived patience (180s base, scaled higher on long contexts). The configured
value already reaches the OpenAI and Anthropic clients; these tests pin that it
reaches both Bedrock client construction paths — the ambient credential chain
and the routed multiplex profile — and that leaving it unset changes nothing.
"""
import pytest

from agent import bedrock_adapter
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


class _FakeBoto3:
    """Records the kwargs each ``.client()`` seam was called with."""

    __version__ = "1.34.99"

    def __init__(self):
        self.seen: dict = {}

    def client(self, service, **kwargs):
        self.seen["ambient"] = kwargs
        return object()

    def Session(self, **kwargs):  # noqa: N802 - mirrors the boto3 API
        outer = self

        class _Session:
            def client(self, service, **client_kwargs):
                outer.seen["routed"] = client_kwargs
                return object()

        return _Session()


def _write_config(home, timeout):
    home.mkdir(parents=True, exist_ok=True)
    body = "providers:\n  bedrock:\n" if timeout is None else (
        f"providers:\n  bedrock:\n    request_timeout_seconds: {timeout}\n")
    (home / "config.yaml").write_text(body, encoding="utf-8")


def _write_zero_config(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "providers:\n  bedrock:\n    request_timeout_seconds: 0\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_client_caches():
    bedrock_adapter.reset_client_cache()
    bedrock_adapter._bedrock_clients_by_home.clear()
    yield
    bedrock_adapter.reset_client_cache()
    bedrock_adapter._bedrock_clients_by_home.clear()


def _ambient_kwargs(tmp_path, timeout, monkeypatch):
    _write_config(tmp_path / "home", timeout)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    fake = _FakeBoto3()
    monkeypatch.setattr(bedrock_adapter, "_require_boto3", lambda: fake)
    bedrock_adapter.reset_client_cache()
    bedrock_adapter._cached_client({}, "bedrock-runtime", "us-east-1")
    return fake.seen["ambient"]


def test_configured_request_timeout_reaches_the_ambient_client(tmp_path, monkeypatch):
    """The headline case: a configured value must become botocore's read_timeout."""
    kwargs = _ambient_kwargs(tmp_path, 300, monkeypatch)
    assert "config" in kwargs, "configured timeout never reached the boto3 client"
    assert kwargs["config"].read_timeout == 300


def test_unset_request_timeout_leaves_botocore_defaults_alone(tmp_path, monkeypatch):
    """No configured value must not invent a Config — botocore's own defaults stand."""
    kwargs = _ambient_kwargs(tmp_path, None, monkeypatch)
    assert "config" not in kwargs
    assert kwargs == {"region_name": "us-east-1"}


def test_zero_request_timeout_is_treated_as_unset(tmp_path, monkeypatch):
    """0 is falsy and ``_coerce_timeout`` maps non-positive to None — stay on defaults."""
    _write_zero_config(tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    fake = _FakeBoto3()
    monkeypatch.setattr(bedrock_adapter, "_require_boto3", lambda: fake)
    bedrock_adapter.reset_client_cache()
    bedrock_adapter._cached_client({}, "bedrock-runtime", "us-east-1")
    assert "config" not in fake.seen["ambient"]


def test_configured_request_timeout_reaches_the_routed_profile_client(tmp_path, monkeypatch):
    """The multiplex path builds its client through ``boto3.Session``, not ``boto3.client``;
    it must carry the same timeout or routed profiles stay pinned at 60s."""
    home = tmp_path / "home"
    _write_config(home, 300)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(bedrock_adapter, "scoped_aws_session_kwargs", lambda: {})
    fake = _FakeBoto3()
    monkeypatch.setattr(bedrock_adapter, "_require_boto3", lambda: fake)
    bedrock_adapter.reset_client_cache()
    bedrock_adapter._bedrock_clients_by_home.clear()

    token = set_hermes_home_override(str(home))
    try:
        bedrock_adapter._cached_client(
            bedrock_adapter._bedrock_runtime_client_cache, "bedrock-runtime", "us-west-2")
    finally:
        reset_hermes_home_override(token)

    assert "routed" in fake.seen, "routed client seam was never reached"
    assert "config" in fake.seen["routed"]
    assert fake.seen["routed"]["config"].read_timeout == 300


def test_botocore_config_is_optional_at_call_time(tmp_path, monkeypatch):
    """botocore ships with boto3, but a stubbed/absent botocore must not break client build."""
    _write_config(tmp_path / "home", 300)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    real_import = __import__

    def _no_botocore_config(name, *args, **kwargs):
        if name == "botocore.config":
            raise ImportError("no botocore.config")
        return real_import(name, *args, **kwargs)

    fake = _FakeBoto3()
    monkeypatch.setattr(bedrock_adapter, "_require_boto3", lambda: fake)
    monkeypatch.setattr("builtins.__import__", _no_botocore_config)
    bedrock_adapter.reset_client_cache()
    bedrock_adapter._cached_client({}, "bedrock-runtime", "us-east-1")
    assert "config" not in fake.seen["ambient"]