from __future__ import annotations

from pathlib import Path

import pytest

from morai_ddv2.cache.manifest import assign_bag_splits, relative_bag_path


EXPECTED_TRAIN = [
    "세션/20.bag",
    "세션/11.bag",
    "세션/24.bag",
    "세션/00.bag",
    "세션/17.bag",
    "세션/04.bag",
    "세션/29.bag",
    "세션/16.bag",
    "세션/10.bag",
    "세션/07.bag",
    "세션/27.bag",
    "세션/08.bag",
    "세션/12.bag",
    "세션/26.bag",
    "세션/19.bag",
    "세션/09.bag",
    "세션/28.bag",
    "세션/05.bag",
    "세션/13.bag",
    "세션/06.bag",
    "세션/22.bag",
    "세션/18.bag",
    "세션/01.bag",
    "세션/14.bag",
]
EXPECTED_VALIDATION = [
    "세션/02.bag",
    "세션/15.bag",
    "세션/03.bag",
    "세션/21.bag",
    "세션/23.bag",
    "세션/25.bag",
]


def test_sha256_split_is_a_deterministic_24_6_bag_partition() -> None:
    paths = ["세션/{:02d}.bag".format(index) for index in reversed(range(30))]

    split = assign_bag_splits(paths)

    assert [path for path, value in split.items() if value == "train"] == EXPECTED_TRAIN
    assert [path for path, value in split.items() if value == "validation"] == (
        EXPECTED_VALIDATION
    )
    assert len(split) == 30
    assert set(EXPECTED_TRAIN).isdisjoint(EXPECTED_VALIDATION)


def test_relative_bag_path_uses_symlink_resolved_regular_utf8_path(tmp_path: Path) -> None:
    real_root = tmp_path / "실제"
    bag = real_root / "세션" / "원본.bag"
    bag.parent.mkdir(parents=True)
    bag.write_bytes(b"bag")
    (real_root / "세션" / "별칭.bag").symlink_to(bag)
    linked_root = tmp_path / "가방-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    assert relative_bag_path(linked_root, linked_root / "세션" / "별칭.bag") == (
        "세션/원본.bag"
    )


def test_relative_bag_path_rejects_symlink_escape_and_non_regular_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bag"
    outside.write_bytes(b"outside")
    (root / "escape.bag").symlink_to(outside)

    with pytest.raises(ValueError, match="outside"):
        relative_bag_path(root, root / "escape.bag")
    with pytest.raises(ValueError, match="regular"):
        relative_bag_path(root, root)


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "../escape.bag",
        "/absolute.bag",
        "windows\\path.bag",
        "double//slash.bag",
        "nul\x00path.bag",
        "bad\ud800unicode.bag",
    ],
)
def test_split_rejects_noncanonical_or_non_utf8_paths(path: str) -> None:
    with pytest.raises((TypeError, ValueError, UnicodeError)):
        assign_bag_splits([path])


def test_split_rejects_duplicate_relative_paths() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        assign_bag_splits(["same.bag", "same.bag"])
