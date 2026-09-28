import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from wealth_advisor.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "sk-test-not-a-real-key"


@pytest.fixture(autouse=True)
def _isolate_from_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LYZR_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)


def test_reads_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LYZR_API_KEY", FAKE_KEY)

    assert Settings().lyzr_api_key.get_secret_value() == FAKE_KEY


def test_model_defaults_to_gpt_4_1_and_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LYZR_API_KEY", FAKE_KEY)
    default = Settings().lyzr_model
    monkeypatch.setenv("LYZR_MODEL", "anthropic/claude-sonnet-4-5")

    assert (default, Settings().lyzr_model) == ("openai/gpt-4.1", "anthropic/claude-sonnet-4-5")


def test_reads_api_key_from_dotenv_and_ignores_unrelated_entries(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"LYZR_API_KEY={FAKE_KEY}\nOTHER_TOOL=1\n", encoding="utf-8")

    assert Settings().lyzr_api_key.get_secret_value() == FAKE_KEY


def test_environment_variable_wins_over_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("LYZR_API_KEY=key-from-file\n", encoding="utf-8")
    monkeypatch.setenv("LYZR_API_KEY", FAKE_KEY)

    assert Settings().lyzr_api_key.get_secret_value() == FAKE_KEY


def test_api_key_never_leaks_into_repr_or_serialised_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LYZR_API_KEY", FAKE_KEY)
    settings = Settings()

    for rendered in (repr(settings), str(settings), settings.model_dump_json()):
        assert FAKE_KEY not in rendered


def test_missing_api_key_fails_fast() -> None:
    with pytest.raises(ValidationError, match="lyzr_api_key"):
        Settings()


def test_unedited_env_template_fails_fast(tmp_path: Path) -> None:
    shutil.copy(REPO_ROOT / ".env.example", tmp_path / ".env")

    with pytest.raises(ValidationError, match="lyzr_api_key"):
        Settings()
