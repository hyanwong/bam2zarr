from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pysam
import pytest
import xarray as xr

from bam2zarr import convert_bam_to_vcfzarr
from bam2zarr.converter import _open_reference_fasta


@dataclass
class FakeMutation:
    derived_state: str


@dataclass
class FakeSite:
    position: float
    ancestral_state: str
    mutations: list[FakeMutation] = field(default_factory=list)


class FakeTreeSequence:
    def __init__(self, sites: list[FakeSite]) -> None:
        self._sites = sites
        self.num_sites = len(sites)

    def sites(self):
        return iter(self._sites)


def _write_reference_fasta(path: Path, contig_name: str, sequence: str) -> None:
    path.write_text(f">{contig_name}\n{sequence}\n", encoding="utf-8")
    pysam.faidx(str(path))


def _write_bam(path: Path, contig_name: str, reference_sequence: str, reads: list[tuple[str, str]]) -> None:
    header = {
        "HD": {"VN": "1.0"},
        "SQ": [{"SN": contig_name, "LN": len(reference_sequence)}],
    }
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        for name, sequence in reads:
            read = pysam.AlignedSegment()
            read.query_name = name
            read.query_sequence = sequence
            read.flag = 0
            read.reference_id = 0
            read.reference_start = 0
            read.mapping_quality = 60
            read.cigar = ((0, len(sequence)),)
            read.query_qualities = pysam.qualitystring_to_array("I" * len(sequence))
            bam.write(read)


def test_convert_bam_to_vcfzarr_discovers_reference_mismatch_sites(tmp_path: Path) -> None:
    reference_sequence = "AAAAA"
    fasta_path = tmp_path / "reference.fa"
    bam_path = tmp_path / "reads.bam"
    output_path = tmp_path / "output.zarr"

    _write_reference_fasta(fasta_path, "chr1", reference_sequence)
    _write_bam(
        bam_path,
        "chr1",
        reference_sequence,
        [
            ("read_c", "ACAAA"),
            ("read_g", "AGAAA"),
            ("read_ref", "AAAAA"),
        ],
    )

    summary = convert_bam_to_vcfzarr(
        bam_path=str(bam_path),
        output_zarr_path=str(output_path),
        reference_fasta_path=str(fasta_path),
    )

    ds = xr.open_zarr(output_path)

    assert summary.sample_count == 3
    assert summary.variant_count == 1
    assert summary.known_sites_count == 0
    assert ds["variant_position"].values.tolist() == [2]
    assert ds["variant_allele"].values[0].tolist() == ["A", "C", "G"]
    assert ds["call_genotype"].values[0, :, 0].tolist() == [1, 2, 0]
    assert all(isinstance(entry, (bytes, bytearray)) for entry in ds["individuals_metadata"].values.tolist())
    assert [
        json.loads(entry)["QNAME"] for entry in ds["individuals_metadata"].values.tolist()
    ] == ["read_c", "read_g", "read_ref"]


def test_convert_bam_to_vcfzarr_accepts_known_sites_path(tmp_path: Path) -> None:
    reference_sequence = "AAAAA"
    fasta_path = tmp_path / "reference.fa"
    bam_path = tmp_path / "reads.bam"
    known_sites_path = tmp_path / "sites.vcf"
    output_path = tmp_path / "output.zarr"

    _write_reference_fasta(fasta_path, "chr1", reference_sequence)
    _write_bam(
        bam_path,
        "chr1",
        reference_sequence,
        [("read_alt", "ATAAA"), ("read_ref", "AAAAA")],
    )
    known_sites_path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=5>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t2\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )

    summary = convert_bam_to_vcfzarr(
        bam_path=str(bam_path),
        output_zarr_path=str(output_path),
        known_sites=str(known_sites_path),
        reference_fasta_path=str(fasta_path),
    )

    ds = xr.open_zarr(output_path)

    assert summary.sample_count == 2
    assert summary.variant_count == 1
    assert summary.known_sites_count == 1
    assert ds["variant_position"].values.tolist() == [2]
    assert ds["variant_allele"].values[0].tolist() == ["A", "T"]
    assert ds["call_genotype"].values[0, :, 0].tolist() == [1, 0]


def test_convert_bam_to_vcfzarr_accepts_tree_sequences(tmp_path: Path) -> None:
    reference_sequence = "AAAAA"
    fasta_path = tmp_path / "reference.fa"
    bam_path = tmp_path / "reads.bam"
    output_path = tmp_path / "output.zarr"

    _write_reference_fasta(fasta_path, "chr1", reference_sequence)
    _write_bam(
        bam_path,
        "chr1",
        reference_sequence,
        [("read_ref", "AAAAA")],
    )

    tree_sequences = {
        "chr1": FakeTreeSequence(
            [
                FakeSite(
                    position=1.0,
                    ancestral_state="A",
                    mutations=[FakeMutation("T")],
                )
            ]
        )
    }

    summary = convert_bam_to_vcfzarr(
        bam_path=str(bam_path),
        output_zarr_path=str(output_path),
        known_sites=tree_sequences,
        tree_sequence_position_base=0,
        reference_fasta_path=str(fasta_path),
    )

    ds = xr.open_zarr(output_path)

    assert summary.sample_count == 1
    assert summary.variant_count == 1
    assert summary.known_sites_count == 1
    assert ds["variant_position"].values.tolist() == [2]
    assert ds["variant_allele"].values[0].tolist() == ["A", "T"]
    assert ds["call_genotype"].values[0, :, 0].tolist() == [0]


def test_convert_bam_to_vcfzarr_accepts_one_based_tree_sequences(tmp_path: Path) -> None:
    reference_sequence = "AAAAA"
    fasta_path = tmp_path / "reference.fa"
    bam_path = tmp_path / "reads.bam"
    output_path = tmp_path / "output.zarr"

    _write_reference_fasta(fasta_path, "chr1", reference_sequence)
    _write_bam(
        bam_path,
        "chr1",
        reference_sequence,
        [("read_ref", "AAAAA")],
    )

    tree_sequences = {
        "chr1": FakeTreeSequence(
            [
                FakeSite(
                    position=2.0,
                    ancestral_state="A",
                    mutations=[FakeMutation("T")],
                )
            ]
        )
    }

    summary = convert_bam_to_vcfzarr(
        bam_path=str(bam_path),
        output_zarr_path=str(output_path),
        known_sites=tree_sequences,
        tree_sequence_position_base=1,
        reference_fasta_path=str(fasta_path),
    )

    ds = xr.open_zarr(output_path)

    assert summary.sample_count == 1
    assert summary.variant_count == 1
    assert summary.known_sites_count == 1
    assert ds["variant_position"].values.tolist() == [2]
    assert ds["variant_allele"].values[0].tolist() == ["A", "T"]
    assert ds["call_genotype"].values[0, :, 0].tolist() == [0]


def test_convert_bam_to_vcfzarr_reports_tree_sequence_reference_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference_sequence = "AAAAA"
    fasta_path = tmp_path / "reference.fa"
    bam_path = tmp_path / "reads.bam"
    output_path = tmp_path / "output.zarr"

    _write_reference_fasta(fasta_path, "chr1", reference_sequence)
    _write_bam(
        bam_path,
        "chr1",
        reference_sequence,
        [("read_ref", "AAAAA")],
    )

    tree_sequences = {
        "chr1": FakeTreeSequence(
            [
                FakeSite(
                    position=2.0,
                    ancestral_state="A",
                    mutations=[FakeMutation("T")],
                )
            ]
        )
    }

    summary = convert_bam_to_vcfzarr(
        bam_path=str(bam_path),
        output_zarr_path=str(output_path),
        known_sites=tree_sequences,
        tree_sequence_position_base=1,
        reference_fasta_path=str(fasta_path),
    )

    captured = capsys.readouterr()

    assert summary.tree_sequence_comparable_site_count == 1
    assert summary.tree_sequence_ancestral_ref_mismatch_site_count == 0
    assert summary.tree_sequence_ancestral_ref_match_proportion == 1.0
    assert summary.tree_sequence_ancestral_ref_mismatch_proportion == 0.0
    assert "Tree-sequence ancestral/reference match: 100.00%" in captured.err
    assert "mismatch: 0.00% (0/1)" in captured.err


def test_convert_bam_to_vcfzarr_reports_tree_sequence_reference_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference_sequence = "AACAA"
    fasta_path = tmp_path / "reference.fa"
    bam_path = tmp_path / "reads.bam"
    output_path = tmp_path / "output.zarr"

    _write_reference_fasta(fasta_path, "chr1", reference_sequence)
    _write_bam(
        bam_path,
        "chr1",
        reference_sequence,
        [("read_ref", "AACAA")],
    )

    tree_sequences = {
        "chr1": FakeTreeSequence(
            [
                FakeSite(
                    position=2.0,
                    ancestral_state="A",
                    mutations=[FakeMutation("T")],
                )
            ]
        )
    }

    summary = convert_bam_to_vcfzarr(
        bam_path=str(bam_path),
        output_zarr_path=str(output_path),
        known_sites=tree_sequences,
        tree_sequence_position_base=0,
        reference_fasta_path=str(fasta_path),
    )

    captured = capsys.readouterr()

    assert summary.tree_sequence_comparable_site_count == 1
    assert summary.tree_sequence_ancestral_ref_mismatch_site_count == 1
    assert summary.tree_sequence_ancestral_ref_match_proportion == 0.0
    assert summary.tree_sequence_ancestral_ref_mismatch_proportion == 1.0
    assert "Tree-sequence ancestral/reference match: 0.00%" in captured.err
    assert "mismatch: 100.00% (1/1)" in captured.err
    assert "off-by-one coordinate error" in captured.err


def test_convert_bam_to_vcfzarr_individuals_metadata_tsinfer_compat(tmp_path: Path) -> None:
    tsinfer = pytest.importorskip("tsinfer")
    if getattr(tsinfer, "__version__", None) != "0.5.1":
        pytest.skip("Compatibility check is pinned to tsinfer==0.5.1")

    reference_sequence = "AAAAA"
    fasta_path = tmp_path / "reference.fa"
    bam_path = tmp_path / "reads.bam"
    output_path = tmp_path / "output.zarr"

    _write_reference_fasta(fasta_path, "chr1", reference_sequence)
    _write_bam(
        bam_path,
        "chr1",
        reference_sequence,
        [
            ("read_c", "ACAAA"),
            ("read_ref", "AAAAA"),
        ],
    )

    convert_bam_to_vcfzarr(
        bam_path=str(bam_path),
        output_zarr_path=str(output_path),
        reference_fasta_path=str(fasta_path),
    )

    vd = tsinfer.VariantData(str(output_path), ancestral_state=np.array(["A"], dtype=str))
    metadata = vd.individuals_metadata

    assert len(metadata) == 2
    assert [row["QNAME"] for row in metadata] == ["read_c", "read_ref"]


def test_open_reference_fasta_reports_bgzf_hint_for_gz_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_faidx(_path: str) -> None:
        raise OSError("indexing failed")

    monkeypatch.setattr(pysam, "faidx", fake_faidx)

    with pytest.raises(ValueError) as exc_info:
        _open_reference_fasta("reference.fa.gz")

    assert "BGZF-compressed" in str(exc_info.value)


def test_convert_bam_to_vcfzarr_saves_extra_positions(tmp_path: Path) -> None:
    reference_sequence = "AAAAA"
    fasta_path = tmp_path / "reference.fa"
    bam_path = tmp_path / "reads.bam"
    known_sites_path = tmp_path / "sites.vcf"
    output_path = tmp_path / "output.zarr"
    extra_positions_path = tmp_path / "extra_positions.tsv"

    _write_reference_fasta(fasta_path, "chr1", reference_sequence)
    _write_bam(
        bam_path,
        "chr1",
        reference_sequence,
        [("read_alt", "ATATA")],
    )
    known_sites_path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=5>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t2\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )

    convert_bam_to_vcfzarr(
        bam_path=str(bam_path),
        output_zarr_path=str(output_path),
        known_sites=str(known_sites_path),
        reference_fasta_path=str(fasta_path),
        save_extra_positions=str(extra_positions_path),
    )

    assert extra_positions_path.read_text(encoding="utf-8") == "chr1\t4\n"


def test_convert_bam_to_vcfzarr_raises_if_output_store_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_sequence = "AAAAA"
    fasta_path = tmp_path / "reference.fa"
    bam_path = tmp_path / "reads.bam"
    output_path = tmp_path / "missing_output.zarr"

    _write_reference_fasta(fasta_path, "chr1", reference_sequence)
    _write_bam(
        bam_path,
        "chr1",
        reference_sequence,
        [("read_ref", "AAAAA")],
    )

    import shutil
    import zarr

    real_consolidate = zarr.consolidate_metadata

    def fake_consolidate(store, *args, **kwargs):
        real_consolidate(store, *args, **kwargs)
        shutil.rmtree(store, ignore_errors=True)

    monkeypatch.setattr(zarr, "consolidate_metadata", fake_consolidate)

    with pytest.raises(FileNotFoundError) as exc_info:
        convert_bam_to_vcfzarr(
            bam_path=str(bam_path),
            output_zarr_path=str(output_path),
            reference_fasta_path=str(fasta_path),
        )

    assert str(output_path) in str(exc_info.value)
