"""Fleet file validation."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nanobot.fleet.config import FleetConfigError, load_fleet


class TestInstanceParsing:
    def test_resolves_config_dir_and_workspace(
        self,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        fleet_path = write_fleet({"research": make_entry("research")})
        fleet = load_fleet(fleet_path)

        instance = fleet.instance("research")
        assert instance is not None
        assert instance.config_dir == instance.config_path.parent
        assert instance.workspace == (instance.config_dir / "workspace").resolve()
        assert instance.mode == "gateway"
        assert instance.memory_limit_mb == 256

    def test_workspace_falls_back_to_the_nanobot_default(
        self,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        entry = make_entry("research", config_data={"agents": {"defaults": {}}})
        fleet = load_fleet(write_fleet({"research": entry}))

        instance = fleet.instance("research")
        assert instance is not None
        assert instance.workspace == (Path.home() / ".nanobot" / "workspace").resolve()

    def test_env_names_are_deduplicated_in_order(
        self,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        entry = make_entry("trader", env=["BROKER_TOKEN", "OPENROUTER_API_KEY", "BROKER_TOKEN"])
        fleet = load_fleet(write_fleet({"trader": entry}))

        instance = fleet.instance("trader")
        assert instance is not None
        assert instance.env == ("BROKER_TOKEN", "OPENROUTER_API_KEY")

    def test_env_defaults_to_empty(
        self,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        fleet = load_fleet(write_fleet({"trader": make_entry("trader")}))

        instance = fleet.instance("trader")
        assert instance is not None
        assert instance.env == ()

    def test_others_excludes_the_named_instance(
        self,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        fleet = load_fleet(
            write_fleet({"a": make_entry("a"), "b": make_entry("b"), "c": make_entry("c")})
        )

        assert {entry.name for entry in fleet.others("b")} == {"a", "c"}
        assert fleet.instance("missing") is None


class TestRejectedFleetFiles:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FleetConfigError, match="cannot be read"):
            load_fleet(tmp_path / "absent.json")

    def test_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "fleet.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(FleetConfigError, match="not valid JSON"):
            load_fleet(path)

    def test_non_object_document(self, tmp_path: Path) -> None:
        path = tmp_path / "fleet.json"
        path.write_text("[]", encoding="utf-8")

        with pytest.raises(FleetConfigError, match="must contain a JSON object"):
            load_fleet(path)

    @pytest.mark.parametrize("document", ['{"instances": {}}', '{"instances": []}', "{}"])
    def test_requires_instances(self, tmp_path: Path, document: str) -> None:
        path = tmp_path / "fleet.json"
        path.write_text(document, encoding="utf-8")

        with pytest.raises(FleetConfigError, match="non-empty 'instances' object"):
            load_fleet(path)

    @pytest.mark.parametrize("name", ["Research", "_research", "-research", "re search", ""])
    def test_rejects_invalid_names(
        self,
        name: str,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        with pytest.raises(FleetConfigError, match="instance name"):
            load_fleet(write_fleet({name: make_entry("research")}))

    def test_rejects_non_object_entry(self, write_fleet: Callable[..., Path]) -> None:
        with pytest.raises(FleetConfigError, match="must be a JSON object"):
            load_fleet(write_fleet({"research": "config.json"}))

    def test_rejects_missing_config(self, write_fleet: Callable[..., Path]) -> None:
        with pytest.raises(FleetConfigError, match="must set 'config'"):
            load_fleet(write_fleet({"research": {"memoryLimitMb": 128}}))

    def test_rejects_absent_config_file(
        self,
        tmp_path: Path,
        write_fleet: Callable[..., Path],
    ) -> None:
        entry = {"config": str(tmp_path / "nope" / "config.json"), "memoryLimitMb": 128}

        with pytest.raises(FleetConfigError, match="config file not found"):
            load_fleet(write_fleet({"research": entry}))

    def test_rejects_unknown_mode(
        self,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        with pytest.raises(FleetConfigError, match="'mode' must be one of"):
            load_fleet(write_fleet({"research": make_entry("research", mode="webui")}))

    @pytest.mark.parametrize("limit", [None, 0, -5, True, "512", 1.5])
    def test_rejects_bad_memory_limit(
        self,
        limit: object,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        entry = make_entry("research")
        entry["memoryLimitMb"] = limit
        if limit is None:
            del entry["memoryLimitMb"]

        with pytest.raises(FleetConfigError, match="'memoryLimitMb' must be a positive integer"):
            load_fleet(write_fleet({"research": entry}))

    @pytest.mark.parametrize("env", ["OPENROUTER_API_KEY", [1], ["not-a-name"]])
    def test_rejects_bad_env(
        self,
        env: object,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        with pytest.raises(FleetConfigError, match="'env'|environment variable name"):
            load_fleet(write_fleet({"research": make_entry("research", env=env)}))

    def test_rejects_unreadable_instance_config(
        self,
        tmp_path: Path,
        write_fleet: Callable[..., Path],
    ) -> None:
        config_path = tmp_path / "broken.json"
        config_path.write_text("{oops", encoding="utf-8")
        entry = {"config": str(config_path), "memoryLimitMb": 128}

        with pytest.raises(FleetConfigError, match="config file is not valid JSON"):
            load_fleet(write_fleet({"research": entry}))

    def test_rejects_non_object_instance_config(
        self,
        tmp_path: Path,
        write_fleet: Callable[..., Path],
    ) -> None:
        config_path = tmp_path / "list.json"
        config_path.write_text("[]", encoding="utf-8")
        entry = {"config": str(config_path), "memoryLimitMb": 128}

        with pytest.raises(FleetConfigError, match="must contain a JSON object"):
            load_fleet(write_fleet({"research": entry}))

    def test_rejects_non_string_workspace(
        self,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        entry = make_entry(
            "research", config_data={"agents": {"defaults": {"workspace": 7}}}
        )

        with pytest.raises(FleetConfigError, match="workspace must be a string"):
            load_fleet(write_fleet({"research": entry}))


class TestOverlapRefusal:
    def test_identical_workspaces_name_both_instances(
        self,
        tmp_path: Path,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        shared = tmp_path / "shared-workspace"
        entries = {
            "research": make_entry("research", workspace=shared),
            "trader": make_entry("trader", workspace=shared),
        }

        with pytest.raises(FleetConfigError) as excinfo:
            load_fleet(write_fleet(entries))

        message = str(excinfo.value)
        assert "research" in message and "trader" in message
        assert "overlap" in message

    def test_nested_workspace_names_both_instances(
        self,
        tmp_path: Path,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        outer = tmp_path / "outer"
        entries = {
            "research": make_entry("research", workspace=outer),
            "trader": make_entry("trader", workspace=outer / "inner"),
        }

        with pytest.raises(FleetConfigError) as excinfo:
            load_fleet(write_fleet(entries))

        message = str(excinfo.value)
        assert "research" in message and "trader" in message
        assert "is inside" in message

    def test_workspace_inside_another_config_directory_is_refused(
        self,
        tmp_path: Path,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        trader_config_dir = tmp_path / ".nanobot-trader"
        entries = {
            "trader": make_entry("trader"),
            "research": make_entry("research", workspace=trader_config_dir / "stolen"),
        }

        with pytest.raises(FleetConfigError) as excinfo:
            load_fleet(write_fleet(entries))

        message = str(excinfo.value)
        assert "research" in message and "trader" in message

    def test_workspace_inside_its_own_config_directory_is_fine(
        self,
        make_entry: Callable[..., dict[str, Any]],
        write_fleet: Callable[..., Path],
    ) -> None:
        # The default nanobot layout nests the workspace under the config dir.
        fleet = load_fleet(write_fleet({"a": make_entry("a"), "b": make_entry("b")}))

        assert {entry.name for entry in fleet.instances} == {"a", "b"}

    def test_shared_config_directory_is_refused(
        self,
        tmp_path: Path,
        write_fleet: Callable[..., Path],
    ) -> None:
        config_dir = tmp_path / "shared-config"
        config_dir.mkdir()
        entries: dict[str, Any] = {}
        for name in ("research", "trader"):
            config_path = config_dir / f"{name}.json"
            config_path.write_text(
                json.dumps(
                    {"agents": {"defaults": {"workspace": str(tmp_path / f"ws-{name}")}}}
                ),
                encoding="utf-8",
            )
            entries[name] = {"config": str(config_path), "memoryLimitMb": 128}

        with pytest.raises(FleetConfigError) as excinfo:
            load_fleet(write_fleet(entries))

        assert "research" in str(excinfo.value) and "trader" in str(excinfo.value)
