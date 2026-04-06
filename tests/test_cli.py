from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bam2zarr import cli


class _Recorder:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_main_accepts_tree_sequence_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree_sequence_path = tmp_path / "chr1.trees"
    tree_sequence_path.write_text("placeholder", encoding="utf-8")

    loaded_tree_sequence = object()

    def fake_import_module(name: str) -> object:
        assert name == "tskit"
        return SimpleNamespace(load=lambda _path: loaded_tree_sequence)

    recorder = _Recorder()

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(cli, "convert_bam_to_vcfzarr", recorder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bam2zarr",
            "reads.bam",
            "reference.fa",
            "output.zarr",
            "--tree-sequence",
            f"chr1={tree_sequence_path}",
            "--tree-sequence-position-base",
            "1",
            "--no-progress",
        ],
    )

    cli.main()

    assert recorder.kwargs is not None
    assert recorder.kwargs["bam_path"] == "reads.bam"
    assert recorder.kwargs["reference_fasta_path"] == "reference.fa"
    assert recorder.kwargs["output_zarr_path"] == "output.zarr"
    assert recorder.kwargs["tree_sequence_position_base"] == 1
    assert recorder.kwargs["show_progress"] is False
    assert recorder.kwargs["known_sites"] == {"chr1": loaded_tree_sequence}


def test_main_rejects_known_sites_and_tree_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_load_tree_sequences", lambda _entries: {"chr1": object()})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bam2zarr",
            "reads.bam",
            "reference.fa",
            "output.zarr",
            "--known-sites",
            "known_sites.vcf",
            "--tree-sequence",
            "chr1=chr1.trees",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert "either --known-sites or --tree-sequence" in str(exc_info.value)


def test_main_defaults_output_to_bam_with_vcz_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()

    monkeypatch.setattr(cli, "_load_tree_sequences", lambda _entries: {})
    monkeypatch.setattr(cli, "convert_bam_to_vcfzarr", recorder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bam2zarr",
            "reads.bam",
            "reference.fa",
            "--known-sites",
            "known_sites.vcf",
        ],
    )

    cli.main()

    assert recorder.kwargs is not None
    assert recorder.kwargs["output_zarr_path"] == "reads.vcz"


def test_main_passes_save_extra_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()

    monkeypatch.setattr(cli, "_load_tree_sequences", lambda _entries: {})
    monkeypatch.setattr(cli, "convert_bam_to_vcfzarr", recorder)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bam2zarr",
            "reads.bam",
            "reference.fa",
            "output.zarr",
            "--known-sites",
            "known_sites.vcf",
            "--save-extra-positions",
            "extra.tsv",
        ],
    )

    cli.main()

    assert recorder.kwargs is not None
    assert recorder.kwargs["save_extra_positions"] == "extra.tsv"
