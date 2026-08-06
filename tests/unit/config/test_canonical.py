from __future__ import annotations

from pathlib import Path

from morai_ddv2.config.canonical import canonical_json_bytes, sha256_bytes, sha256_file


def test_canonical_json_bytes_are_rfc_8785_stable() -> None:
    first = {"z": 3, "a": [True, None, "é"], "n": 1.0}
    second = {"n": 1.0, "a": [True, None, "é"], "z": 3}

    expected = b'{"a":[true,null,"\xc3\xa9"],"n":1,"z":3}'
    assert canonical_json_bytes(first) == expected
    assert canonical_json_bytes(second) == expected


def test_sha256_helpers_return_lowercase_hex(tmp_path: Path) -> None:
    payload = b"abc"
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    assert sha256_bytes(payload) == expected
    assert sha256_file(path, chunk_size=2) == expected
    assert expected == expected.lower()
