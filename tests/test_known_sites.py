from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from bam2zarr import (
    load_known_sites_auto,
    load_known_sites_tree_sequences,
    load_known_sites_vcf,
    load_known_sites_vcfzarr,
)


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


def test_load_known_sites_vcfzarr(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sites.zarr"
    ds = xr.Dataset(
        data_vars={
            "contig_id": (("contigs",), np.array(["chr1", "chr2"], dtype=object)),
            "variant_contig": (("variants",), np.array([0, 0, 1], dtype=np.int32)),
            "variant_position": (("variants",), np.array([10, 10, 20], dtype=np.int32)),
            "variant_allele": (
                ("variants", "alleles"),
                np.array(
                    [
                        ["A", "C"],
                        ["A", "G"],
                        ["T", "."],
                    ],
                    dtype=object,
                ),
            ),
        }
    )
    ds.to_zarr(zarr_path, mode="w")

    sites = load_known_sites_vcfzarr(str(zarr_path))

    assert sites[("chr1", 10)] == ["A", "C", "G"]
    assert sites[("chr2", 20)] == ["T"]


def test_load_known_sites_vcf(tmp_path: Path) -> None:
    vcf_path = tmp_path / "sites.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t10\t.\tA\tC,G\t.\tPASS\t.\n"
        "chr1\t20\t.\tT\tA\t.\tPASS\t.\n",
        encoding="utf-8",
    )

    sites = load_known_sites_vcf(str(vcf_path))

    assert sites[("chr1", 10)] == ["A", "C", "G"]
    assert sites[("chr1", 20)] == ["T", "A"]


def test_load_known_sites_auto_uses_vcf_loader(tmp_path: Path) -> None:
    vcf_path = tmp_path / "sites.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t30\t.\tC\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )

    sites = load_known_sites_auto(str(vcf_path))

    assert sites[("chr1", 30)] == ["C", "T"]


def test_load_known_sites_tree_sequences() -> None:
    tree_sequences = {
        "chr1": FakeTreeSequence(
            [
                FakeSite(
                    position=9.0,
                    ancestral_state="A",
                    mutations=[FakeMutation("C"), FakeMutation("G"), FakeMutation("C")],
                )
            ]
        )
    }

    sites = load_known_sites_tree_sequences(tree_sequences)

    assert sites[("chr1", 10)] == ["A", "C", "G"]


def test_load_known_sites_tree_sequences_one_based() -> None:
    tree_sequences = {
        "chr1": FakeTreeSequence(
            [
                FakeSite(
                    position=10.0,
                    ancestral_state="A",
                    mutations=[FakeMutation("C"), FakeMutation("G")],
                )
            ]
        )
    }

    sites = load_known_sites_tree_sequences(tree_sequences, position_base=1)

    assert sites[("chr1", 10)] == ["A", "C", "G"]
