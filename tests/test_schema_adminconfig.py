"""Unit tests for AdminConfig schema serialization and GatewayConfig integration."""

import pytest

from nanobot.config.schema import AdminConfig, GatewayConfig, MatrixConfig


# ---------------------------------------------------------------------------
# AdminConfig — defaults
# ---------------------------------------------------------------------------


def test_admin_config_defaults():
    """AdminConfig should have secure defaults (disabled, no token, localhost)."""
    admin = AdminConfig()
    assert admin.enabled is False
    assert admin.token == ""
    assert admin.host == "127.0.0.1"
    assert admin.port == 18791


# ---------------------------------------------------------------------------
# AdminConfig — snake_case round-trip
# ---------------------------------------------------------------------------


def test_admin_config_snake_case_round_trip():
    """AdminConfig should serialize and deserialize via snake_case field names."""
    original = AdminConfig(enabled=True, token="secret", host="0.0.0.0", port=9000)

    dumped = original.model_dump()
    restored = AdminConfig(**dumped)

    assert restored.enabled is True
    assert restored.token == "secret"
    assert restored.host == "0.0.0.0"
    assert restored.port == 9000


# ---------------------------------------------------------------------------
# AdminConfig — camelCase round-trip
# ---------------------------------------------------------------------------


def test_admin_config_camel_case_parse():
    """AdminConfig should accept camelCase keys (alias_generator=to_camel)."""
    data = {
        "enabled": True,
        "token": "bearer-xyz",
        "host": "192.168.1.1",
        "port": 8080,
    }
    admin = AdminConfig.model_validate(data)

    assert admin.enabled is True
    assert admin.token == "bearer-xyz"
    assert admin.host == "192.168.1.1"
    assert admin.port == 8080


def test_admin_config_camel_case_serialise():
    """AdminConfig.model_dump(by_alias=True) should produce camelCase keys."""
    admin = AdminConfig(enabled=True, token="tok", host="127.0.0.1", port=18791)
    dumped = admin.model_dump(by_alias=True)

    assert "enabled" in dumped  # 'enabled' has no camelCase equivalent change
    assert "token" in dumped
    assert "host" in dumped
    assert "port" in dumped


def test_admin_config_camel_case_field_names():
    """Fields with snake_case names that differ in camelCase should round-trip."""
    # AdminConfig has no multi-word field names currently, but we still verify
    # that the round-trip through by_alias dumps is lossless.
    admin = AdminConfig(enabled=False, token="", host="127.0.0.1", port=18791)
    by_alias = admin.model_dump(by_alias=True)
    restored = AdminConfig.model_validate(by_alias)

    assert restored == admin


# ---------------------------------------------------------------------------
# GatewayConfig — includes admin field
# ---------------------------------------------------------------------------


def test_gateway_config_has_admin_field():
    """GatewayConfig should expose an admin sub-config."""
    gw = GatewayConfig()
    assert hasattr(gw, "admin")
    assert isinstance(gw.admin, AdminConfig)


def test_gateway_config_admin_defaults():
    """GatewayConfig.admin should be disabled by default."""
    gw = GatewayConfig()
    assert gw.admin.enabled is False
    assert gw.admin.port == 18791


def test_gateway_config_admin_does_not_collide_with_gateway_port():
    """Admin port default (18791) must differ from gateway port default (18790)."""
    gw = GatewayConfig()
    assert gw.admin.port != gw.port


def test_gateway_config_snake_case_round_trip():
    """GatewayConfig with nested AdminConfig should survive a snake_case round-trip."""
    gw = GatewayConfig(
        host="0.0.0.0",
        port=18790,
        admin=AdminConfig(enabled=True, token="abc", host="127.0.0.1", port=18791),
    )
    dumped = gw.model_dump()
    restored = GatewayConfig(**dumped)

    assert restored.admin.enabled is True
    assert restored.admin.token == "abc"
    assert restored.host == "0.0.0.0"


def test_gateway_config_camel_case_round_trip():
    """GatewayConfig should accept camelCase nested keys."""
    data = {
        "host": "0.0.0.0",
        "port": 18790,
        "admin": {
            "enabled": True,
            "token": "my-secret",
            "host": "127.0.0.1",
            "port": 18791,
        },
    }
    gw = GatewayConfig.model_validate(data)

    assert gw.admin.enabled is True
    assert gw.admin.token == "my-secret"


# ---------------------------------------------------------------------------
# MatrixConfig — snake_case round-trip
# ---------------------------------------------------------------------------


def test_matrix_config_defaults():
    """MatrixConfig should have sensible defaults."""
    mx = MatrixConfig()
    assert mx.enabled is False
    assert mx.homeserver == "https://matrix.org"
    assert mx.e2ee_enabled is True
    assert mx.group_policy == "open"
    assert mx.allow_room_mentions is False


def test_matrix_config_snake_case_round_trip():
    """MatrixConfig should serialise and deserialise via snake_case field names."""
    original = MatrixConfig(
        enabled=True,
        homeserver="https://example.org",
        access_token="tok",
        user_id="@bot:example.org",
        device_id="DEVICE1",
        e2ee_enabled=False,
        sync_stop_grace_seconds=5,
        max_media_bytes=1024,
        allow_from=["@alice:example.org"],
        group_policy="mention",
        group_allow_from=["!room:example.org"],
        allow_room_mentions=True,
    )
    dumped = original.model_dump()
    restored = MatrixConfig(**dumped)

    assert restored.enabled is True
    assert restored.homeserver == "https://example.org"
    assert restored.e2ee_enabled is False
    assert restored.sync_stop_grace_seconds == 5
    assert restored.max_media_bytes == 1024
    assert restored.allow_from == ["@alice:example.org"]
    assert restored.group_policy == "mention"
    assert restored.group_allow_from == ["!room:example.org"]
    assert restored.allow_room_mentions is True


# ---------------------------------------------------------------------------
# MatrixConfig — camelCase round-trip
# ---------------------------------------------------------------------------


def test_matrix_config_camel_case_parse():
    """MatrixConfig should accept camelCase keys produced by alias_generator.

    Note: pydantic's to_camel capitalizes only the first character of each
    underscore-separated segment, so 'e2ee_enabled' → 'e2EeEnabled' (not
    'e2eeEnabled').
    """
    data = {
        "enabled": True,
        "homeserver": "https://matrix.org",
        "accessToken": "tok123",
        "userId": "@bot:matrix.org",
        "deviceId": "DEV",
        "e2EeEnabled": False,   # pydantic to_camel: 'e2ee' → 'e2Ee'
        "syncStopGraceSeconds": 3,
        "maxMediaBytes": 512,
        "allowFrom": [],
        "groupPolicy": "open",
        "groupAllowFrom": [],
        "allowRoomMentions": False,
    }
    mx = MatrixConfig.model_validate(data)

    assert mx.enabled is True
    assert mx.access_token == "tok123"
    assert mx.user_id == "@bot:matrix.org"
    assert mx.e2ee_enabled is False
    assert mx.sync_stop_grace_seconds == 3


def test_matrix_config_camel_case_serialise_and_restore():
    """MatrixConfig serialized with by_alias=True should deserialise losslessly."""
    original = MatrixConfig(
        enabled=True,
        homeserver="https://matrix.org",
        access_token="abc",
        user_id="@bot:matrix.org",
        device_id="DEV",
        e2ee_enabled=True,
        sync_stop_grace_seconds=2,
        max_media_bytes=20971520,
        allow_from=[],
        group_policy="open",
        group_allow_from=[],
        allow_room_mentions=False,
    )
    by_alias = original.model_dump(by_alias=True)

    # by_alias keys should be camelCase for multi-word fields.
    # pydantic to_camel: 'e2ee_enabled' → 'e2EeEnabled' (capitalises first char of each segment).
    assert "accessToken" in by_alias
    assert "userId" in by_alias
    assert "e2EeEnabled" in by_alias
    assert "syncStopGraceSeconds" in by_alias
    assert "maxMediaBytes" in by_alias

    # Validate that the camelCase dump can be round-tripped back
    restored = MatrixConfig.model_validate(by_alias)
    assert restored == original
