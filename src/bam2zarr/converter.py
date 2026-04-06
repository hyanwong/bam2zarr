"""Core BAM to VCF Zarr conversion logic."""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pysam
import xarray as xr
import zarr
import numcodecs


PositionKey = Tuple[int, int]  # (contig_index, 1-based position)
SiteKey = Tuple[int, int]  # (contig_index, 1-based position)
KnownSites = Dict[Tuple[str, int], List[str]]  # (contig_name, 1-based position) -> alleles, first is REF
TreeSequencesByContig = Mapping[str, object]
KnownSitesInput = TreeSequencesByContig | str | PathLike[str]
OutputPathLike = str | PathLike[str]


@dataclass
class ConversionSummary:
    sample_count: int
    variant_count: int
    contig_count: int
    known_sites_count: int
    out_of_known_alleles_site_proportion: float
    tree_sequence_comparable_site_count: int = 0
    tree_sequence_ancestral_ref_mismatch_site_count: int = 0
    tree_sequence_ancestral_ref_match_proportion: float | None = None
    tree_sequence_ancestral_ref_mismatch_proportion: float | None = None


@dataclass(frozen=True)
class TreeSequenceReferenceSummary:
    comparable_site_count: int = 0
    ancestral_ref_mismatch_site_count: int = 0

    @property
    def ancestral_ref_match_proportion(self) -> float | None:
        if self.comparable_site_count == 0:
            return None
        return 1.0 - self.ancestral_ref_mismatch_proportion

    @property
    def ancestral_ref_mismatch_proportion(self) -> float | None:
        if self.comparable_site_count == 0:
            return None
        return (
            float(self.ancestral_ref_mismatch_site_count)
            / float(self.comparable_site_count)
        )


def _unique_sample_name(base_name: str, seen: Dict[str, int]) -> str:
    count = seen[base_name]
    seen[base_name] += 1
    if count == 0:
        return base_name
    return f"{base_name}_{count}"


def _merge_alleles_in_order(existing: List[str], new_alleles: List[str]) -> None:
    for allele in new_alleles:
        if allele not in existing:
            existing.append(allele)


def load_known_sites_file(path: str) -> KnownSites:
    """Load known variable sites from a delimited text file.

    Expected row format:
    - chromosome, position, alleles
    where alleles is comma-separated (for example: A,C,G)
    and the first allele is treated as REF.

    Header rows are allowed.
    """

    known_sites: KnownSites = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
        handle.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            if len(row) < 3:
                continue

            chrom = row[0].strip()
            pos_field = row[1].strip()
            if chrom.lower() in {"chrom", "chromosome", "contig"}:
                continue
            try:
                position = int(pos_field)
            except ValueError:
                continue

            alleles: List[str] = []
            for allele in row[2].split(","):
                a = allele.strip().upper()
                if a not in {"A", "C", "G", "T"}:
                    continue
                if a not in alleles:
                    alleles.append(a)
            if not alleles:
                continue

            key = (chrom, position)
            if key not in known_sites:
                known_sites[key] = alleles
            else:
                _merge_alleles_in_order(known_sites[key], alleles)

    return known_sites


def _normalize_allele_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip().upper()
    if text in {"", ".", "N", "NAN", "NONE"}:
        return None
    if text in {"A", "C", "G", "T"}:
        return text
    return None


def load_known_sites_vcfzarr(path: str) -> KnownSites:
    """Load known variable sites from another VCF Zarr dataset.

    Required variables: contig_id, variant_contig, variant_position, variant_allele.
    """

    ds = xr.open_zarr(path)
    contig_ids = [str(x) for x in ds["contig_id"].values.tolist()]
    variant_contigs = ds["variant_contig"].values
    variant_positions = ds["variant_position"].values
    variant_alleles = ds["variant_allele"].values

    known_sites: KnownSites = {}
    for idx in range(len(variant_positions)):
        contig_index = int(variant_contigs[idx])
        if contig_index < 0 or contig_index >= len(contig_ids):
            continue
        contig_name = contig_ids[contig_index]
        position = int(variant_positions[idx])
        key = (contig_name, position)

        row_alleles: List[str] = []
        for allele in variant_alleles[idx]:
            normalized = _normalize_allele_value(allele)
            if normalized is not None and normalized not in row_alleles:
                row_alleles.append(normalized)
        if not row_alleles:
            continue

        if key not in known_sites:
            known_sites[key] = row_alleles
        else:
            _merge_alleles_in_order(known_sites[key], row_alleles)

    return known_sites


def load_known_sites_vcf(path: str) -> KnownSites:
    """Load known variable sites from a VCF/BCF file.

    Allele order is preserved exactly as in each record: REF first, then ALT list.
    """

    known_sites: KnownSites = {}
    with pysam.VariantFile(path) as vcf:
        for record in vcf.fetch():
            chrom = str(record.contig)
            position = int(record.pos)
            key = (chrom, position)

            row_alleles: List[str] = []
            normalized_ref = _normalize_allele_value(record.ref)
            if normalized_ref is not None:
                row_alleles.append(normalized_ref)

            for alt in record.alts or ():
                normalized_alt = _normalize_allele_value(alt)
                if normalized_alt is not None and normalized_alt not in row_alleles:
                    row_alleles.append(normalized_alt)

            if not row_alleles:
                continue

            if key not in known_sites:
                known_sites[key] = row_alleles
            else:
                _merge_alleles_in_order(known_sites[key], row_alleles)

    return known_sites


def _normalize_tree_sequence_position(position: object, position_base: int) -> int:
    if position_base not in {0, 1}:
        raise ValueError(f"tree_sequence_position_base must be 0 or 1, got {position_base!r}")
    value = float(position)
    rounded = round(value)
    if not np.isfinite(value) or abs(value - rounded) > 1e-9:
        raise ValueError(f"Tree-sequence site positions must be integer-like, got {position!r}")
    return int(rounded) + (1 - position_base)


def _get_reference_base_from_contig_cache(
    reference_fasta: pysam.FastaFile,
    contig_name: str,
    pos1: int,
    contig_sequence_cache: Dict[str, str | None],
) -> str | None:
    if contig_name not in contig_sequence_cache:
        try:
            contig_sequence = reference_fasta.fetch(contig_name)
        except (KeyError, ValueError):
            contig_sequence = ""
        contig_sequence_cache[contig_name] = contig_sequence.upper() if contig_sequence else None

    contig_sequence = contig_sequence_cache[contig_name]
    if contig_sequence is None or pos1 < 1 or pos1 > len(contig_sequence):
        return None

    return _normalize_allele_value(contig_sequence[pos1 - 1])


def _collect_tree_sequence_known_sites(
    tree_sequences: TreeSequencesByContig,
    *,
    position_base: int = 0,
    reference_fasta: pysam.FastaFile | None = None,
    show_progress: bool = False,
) -> tuple[KnownSites, TreeSequenceReferenceSummary]:
    known_sites: KnownSites = {}
    comparable_site_count = 0
    ancestral_ref_mismatch_site_count = 0
    contig_sequence_cache: Dict[str, str | None] = {}

    for chrom, tree_sequence in tree_sequences.items():
        contig_site_total = tree_sequence.num_sites
        total_known = isinstance(contig_site_total, int) and contig_site_total >= 0
        processed_sites = 0
        last_progress_update = 0.0

        sites_method = tree_sequence.sites

        if show_progress:
            _print_tree_sequence_progress(
                contig_name=chrom,
                processed=0,
                total=contig_site_total if total_known else None,
            )

        for site in sites_method():
            processed_sites += 1
            position = _normalize_tree_sequence_position(site.position, position_base)
            key = (chrom, position)

            row_alleles: List[str] = []
            normalized_ancestral = _normalize_allele_value(site.ancestral_state)
            if normalized_ancestral is not None:
                row_alleles.append(normalized_ancestral)

                if reference_fasta is not None:
                    normalized_reference = _get_reference_base_from_contig_cache(
                        reference_fasta,
                        chrom,
                        position,
                        contig_sequence_cache,
                    )
                    if normalized_reference is not None:
                        comparable_site_count += 1
                        if normalized_reference != normalized_ancestral:
                            ancestral_ref_mismatch_site_count += 1

            for mutation in site.mutations:
                normalized_derived = _normalize_allele_value(mutation.derived_state)
                if normalized_derived is not None and normalized_derived not in row_alleles:
                    row_alleles.append(normalized_derived)

            if not row_alleles:
                continue

            if key not in known_sites:
                known_sites[key] = row_alleles
            else:
                _merge_alleles_in_order(known_sites[key], row_alleles)

            if show_progress:
                now = time.monotonic()
                is_final = bool(total_known and processed_sites == contig_site_total)
                if is_final or now - last_progress_update >= 0.2:
                    _print_tree_sequence_progress(
                        contig_name=chrom,
                        processed=processed_sites,
                        total=contig_site_total if total_known else None,
                    )
                    last_progress_update = now

        if show_progress:
            _print_tree_sequence_progress(
                contig_name=chrom,
                processed=processed_sites,
                total=contig_site_total if total_known else None,
            )
            print(file=sys.stderr, flush=True)

    return known_sites, TreeSequenceReferenceSummary(
        comparable_site_count=comparable_site_count,
        ancestral_ref_mismatch_site_count=ancestral_ref_mismatch_site_count,
    )


def load_known_sites_tree_sequences(
    tree_sequences: TreeSequencesByContig, *, position_base: int = 0
) -> KnownSites:
    """Load known variable sites from per-contig tree sequences.

    Tree-sequence site positions can be interpreted as 0-based (tskit-style) or
    1-based genomic coordinates and are normalized to 1-based output positions.
    """

    known_sites, _ = _collect_tree_sequence_known_sites(
        tree_sequences,
        position_base=position_base,
    )
    return known_sites


def load_known_sites_auto(path: str) -> KnownSites:
    """Auto-detect known-sites input format and load it.

    Supported sources:
    - Delimited text file: chromosome,position,alleles
    - VCF/BCF (.vcf, .vcf.gz, .bcf)
    - VCF Zarr store (directory with .zarr suffix or zarr metadata)
    """

    path_obj = Path(path)
    lower_name = path_obj.name.lower()
    suffixes = [s.lower() for s in path_obj.suffixes]

    if path_obj.is_dir() and (
        lower_name.endswith(".zarr")
        or (path_obj / ".zgroup").exists()
        or (path_obj / "zarr.json").exists()
    ):
        return load_known_sites_vcfzarr(path)

    if lower_name.endswith(".vcf") or lower_name.endswith(".vcf.gz") or lower_name.endswith(".bcf"):
        return load_known_sites_vcf(path)

    if ".vcf" in suffixes or ".bcf" in suffixes:
        return load_known_sites_vcf(path)

    return load_known_sites_file(path)


def _looks_like_tree_sequence_map(value: object) -> bool:
    if not isinstance(value, Mapping) or len(value) == 0:
        return False
    return all(isinstance(key, str) for key in value)


def _merge_known_sites_sources(*sources: KnownSites | None) -> KnownSites:
    merged: KnownSites = {}
    for source in sources:
        if source is None:
            continue
        for key, alleles in source.items():
            if key not in merged:
                merged[key] = list(alleles)
            else:
                _merge_alleles_in_order(merged[key], list(alleles))
    return merged


def _load_known_sites_input(
    known_sites: KnownSitesInput | None, *, tree_sequence_position_base: int
) -> KnownSites:
    if known_sites is None:
        return {}

    if isinstance(known_sites, (str, PathLike)):
        return load_known_sites_auto(str(known_sites))

    if _looks_like_tree_sequence_map(known_sites):
        return load_known_sites_tree_sequences(
            known_sites, position_base=tree_sequence_position_base
        )

    raise TypeError(
        "known_sites must be a path-like known-sites source or a dict of tree sequences"
    )


def _open_reference_fasta(path: str) -> pysam.FastaFile:
    fai_path = Path(f"{path}.fai")
    try:
        if not fai_path.exists():
            pysam.faidx(path)
        return pysam.FastaFile(path)
    except (OSError, ValueError) as exc:
        lower_path = str(path).lower()
        if lower_path.endswith(".gz"):
            raise ValueError(
                "Unable to open reference FASTA. For .gz inputs, FASTA must be "
                "BGZF-compressed so it can be indexed by faidx."
            ) from exc
        raise ValueError(
            "Unable to open reference FASTA; ensure the file is readable and faidx-indexable."
        ) from exc


def _count_bam_records(path: str) -> int | None:
    try:
        with pysam.AlignmentFile(path, "rb") as bam:
            return int(bam.count(until_eof=True))
    except (OSError, TypeError, ValueError):
        try:
            with pysam.AlignmentFile(path, "rb") as bam:
                return sum(1 for _ in bam.fetch(until_eof=True))
        except (OSError, ValueError):
            return None


def _emit_tree_sequence_reference_summary(
    summary: TreeSequenceReferenceSummary,
    *,
    position_base: int,
) -> None:
    if summary.comparable_site_count == 0:
        print(
            "Tree-sequence ancestral/reference match: unavailable "
            f"(0 comparable sites, tree_sequence_position_base={position_base})",
            file=sys.stderr,
            flush=True,
        )
        return

    match_proportion = summary.ancestral_ref_match_proportion
    mismatch_proportion = summary.ancestral_ref_mismatch_proportion
    assert match_proportion is not None
    assert mismatch_proportion is not None

    message = (
        "Tree-sequence ancestral/reference match: "
        f"{match_proportion * 100:.2f}% "
        f"({summary.comparable_site_count - summary.ancestral_ref_mismatch_site_count}/"
        f"{summary.comparable_site_count}); mismatch: "
        f"{mismatch_proportion * 100:.2f}% "
        f"({summary.ancestral_ref_mismatch_site_count}/{summary.comparable_site_count}) "
        f"with tree_sequence_position_base={position_base}"
    )
    suggested_base = 1 - position_base
    if 0.2 <= mismatch_proportion <= 0.3:
        message += (
            "; mismatch around 25% often indicates an off-by-one coordinate error; "
            f"try tree_sequence_position_base={suggested_base}"
        )
    elif mismatch_proportion >= 0.1:
        message += (
            "; high mismatch can indicate an off-by-one coordinate error; "
            f"try tree_sequence_position_base={suggested_base}"
        )

    print(message, file=sys.stderr, flush=True)


def _print_bam_progress(
    processed: int,
    total: int,
    *,
    kept_reads: int,
    label: str = "Reading BAM",
) -> None:
    width = 24
    fraction = 0.0 if total == 0 else min(1.0, float(processed) / float(total))
    filled = int(round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r{label} [{bar}] {processed}/{total} {fraction * 100:6.2f}% kept={kept_reads}",
        end="",
        file=sys.stderr,
        flush=True,
    )


def _print_tree_sequence_progress(
    *,
    contig_name: str,
    processed: int,
    total: int | None,
) -> None:
    prefix = f"Reading expected variable sites from tree sequence ({contig_name})"
    if total is None:
        message = f"\r{prefix}: {processed}"
    else:
        width = 24
        fraction = 0.0 if total == 0 else min(1.0, float(processed) / float(total))
        filled = int(round(width * fraction))
        bar = "#" * filled + "-" * (width - filled)
        message = (
            f"\r{prefix} [{bar}] "
            f"{processed}/{total} {fraction * 100:6.2f}%"
        )
    print(message, end="", file=sys.stderr, flush=True)


def _write_extra_positions(
    path: OutputPathLike,
    *,
    sites: List[SiteKey],
    seeded_site_keys: set[PositionKey],
    contig_names: List[str],
) -> None:
    output_path = Path(path)
    extra_sites = [site for site in sites if site not in seeded_site_keys]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for contig_index, position in extra_sites:
            handle.write(f"{contig_names[contig_index]}\t{position}\n")


def convert_bam_to_vcfzarr(
    bam_path: str,
    output_zarr_path: str,
    reference_fasta_path: str,
    known_sites: KnownSitesInput | None = None,
    save_extra_positions: OutputPathLike | None = None,
    tree_sequence_position_base: int = 1,
    min_mapq: int = 0,
    include_secondary: bool = False,
    show_progress: bool = True,
) -> ConversionSummary:
    """Convert each BAM read into a sample in a VCF Zarr-compatible dataset.

    Notes:
    - Each read is represented as one sample.
    - Variant rows are seeded from known_sites.
    - known_sites may be a known-sites path or a per-chromosome tree-sequence mapping.
        - save_extra_positions writes output variant sites that were not present in the
            known-sites input as tab-separated contig and 1-based position pairs.
    - Tree-sequence positions are interpreted according to tree_sequence_position_base.
    - reference_fasta_path is required.
        - show_progress controls BAM progress-bar output on stderr.
    - Any observed fragment base that differs from the reference creates or extends an
      output variant row at that position.
    - The FASTA base is used as REF.
    - Genotypes are haploid encoded as 0 (reference), 1 (alt), or -1 (missing).
    """

    reference_fasta = _open_reference_fasta(reference_fasta_path)
    tree_sequence_reference_summary = TreeSequenceReferenceSummary()

    try:
        if _looks_like_tree_sequence_map(known_sites):
            loaded_known_sites, tree_sequence_reference_summary = _collect_tree_sequence_known_sites(
                known_sites,
                position_base=tree_sequence_position_base,
                reference_fasta=reference_fasta,
                show_progress=show_progress and sys.stderr.isatty(),
            )
            _emit_tree_sequence_reference_summary(
                tree_sequence_reference_summary,
                position_base=tree_sequence_position_base,
            )
            merged_known_sites = _merge_known_sites_sources(loaded_known_sites)
        else:
            merged_known_sites = _merge_known_sites_sources(
                _load_known_sites_input(
                    known_sites, tree_sequence_position_base=tree_sequence_position_base
                ),
            )

        bam_progress_enabled = show_progress and sys.stderr.isatty()
        total_reads = _count_bam_records(bam_path) if bam_progress_enabled else None
        bam = pysam.AlignmentFile(bam_path, "rb")

        contig_names = list(bam.references)
        contig_to_index = {name: idx for idx, name in enumerate(contig_names)}

        sample_ids: List[str] = []
        individuals_metadata: List[bytes] = []
        sample_seen: Dict[str, int] = defaultdict(int)
        known_sites_by_index: Dict[PositionKey, List[str]] = {}
        known_allele_sets: Dict[PositionKey, set[str]] = {}
        known_sites_with_out_of_set_bases: set[PositionKey] = set()
        contig_sequence_cache: Dict[str, str | None] = {}
        seeded_site_keys: set[PositionKey] = set()

        def get_reference_base(contig_name: str, pos1: int) -> str | None:
            contig_index = contig_to_index.get(contig_name)
            if contig_index is None:
                return None
            return _get_reference_base_from_contig_cache(
                reference_fasta,
                contig_name,
                pos1,
                contig_sequence_cache,
            )

        def ensure_site(contig_name: str, pos1: int) -> PositionKey | None:
            contig_index = contig_to_index.get(contig_name)
            if contig_index is None:
                return None
            key = (contig_index, pos1)
            if key not in known_sites_by_index:
                known_sites_by_index[key] = []
                known_allele_sets[key] = set()

            reference_base = get_reference_base(contig_name, pos1)
            if reference_base is not None and reference_base not in known_allele_sets[key]:
                known_sites_by_index[key].insert(0, reference_base)
                known_allele_sets[key].add(reference_base)
            return key

        def add_site_alleles(contig_name: str, pos1: int, alleles: List[str], *, seeded: bool) -> None:
            key = ensure_site(contig_name, pos1)
            if key is None:
                return
            for allele in alleles:
                normalized = _normalize_allele_value(allele)
                if normalized is None or normalized in known_allele_sets[key]:
                    continue
                known_sites_by_index[key].append(normalized)
                known_allele_sets[key].add(normalized)
            if seeded:
                seeded_site_keys.add(key)

        for (contig_name, pos1), alleles in merged_known_sites.items():
            add_site_alleles(contig_name, pos1, alleles, seeded=True)

        known_sites_count = len(seeded_site_keys)

        seeded_allele_sets = {
            key: set(known_sites_by_index[key])
            for key in seeded_site_keys
            if key in known_sites_by_index
        }

        last_progress_update = 0.0
        if total_reads is not None:
            _print_bam_progress(
                0,
                total_reads,
                kept_reads=0,
                label="Reading BAM (pass 1/2)",
            )

        for processed_reads, read in enumerate(bam.fetch(until_eof=True), start=1):
            if read.is_unmapped or read.mapping_quality < min_mapq:
                if total_reads is not None:
                    now = time.monotonic()
                    if (
                        processed_reads == total_reads
                        or now - last_progress_update >= 0.2
                    ):
                        _print_bam_progress(
                            processed_reads,
                            total_reads,
                            kept_reads=len(sample_ids),
                            label="Reading BAM (pass 1/2)",
                        )
                        last_progress_update = now
                continue
            if not include_secondary and (read.is_secondary or read.is_supplementary):
                if total_reads is not None:
                    now = time.monotonic()
                    if (
                        processed_reads == total_reads
                        or now - last_progress_update >= 0.2
                    ):
                        _print_bam_progress(
                            processed_reads,
                            total_reads,
                            kept_reads=len(sample_ids),
                            label="Reading BAM (pass 1/2)",
                        )
                        last_progress_update = now
                continue

            base_name = read.query_name or f"read_{len(sample_ids)}"
            sample_name = _unique_sample_name(base_name, sample_seen)
            sample_ids.append(sample_name)
            individuals_metadata.append(json.dumps({"QNAME": base_name}).encode("utf-8"))

            query_sequence = read.query_sequence or ""

            for query_pos, ref_pos in read.get_aligned_pairs(matches_only=True):
                if query_pos is None or ref_pos is None:
                    continue
                if ref_pos < 0 or ref_pos >= bam.lengths[read.reference_id]:
                    continue

                read_base = query_sequence[query_pos].upper()
                if read_base not in {"A", "C", "G", "T"}:
                    continue

                contig_name = read.reference_name
                if contig_name is None:
                    continue
                pos1 = ref_pos + 1
                reference_base = get_reference_base(contig_name, pos1)

                if reference_base is not None and read_base != reference_base:
                    key = ensure_site(contig_name, pos1)
                    if key is not None and read_base not in known_allele_sets[key]:
                        known_sites_by_index[key].append(read_base)
                        known_allele_sets[key].add(read_base)

                contig_index = contig_to_index[contig_name]
                key = (contig_index, pos1)
                if key not in known_sites_by_index:
                    continue

                seeded_alleles = seeded_allele_sets.get(key)
                if seeded_alleles is not None and read_base not in seeded_alleles:
                    known_sites_with_out_of_set_bases.add(key)

            if total_reads is not None:
                now = time.monotonic()
                if processed_reads == total_reads or now - last_progress_update >= 0.2:
                    _print_bam_progress(
                        processed_reads,
                        total_reads,
                        kept_reads=len(sample_ids),
                        label="Reading BAM (pass 1/2)",
                    )
                    last_progress_update = now

        if total_reads is not None:
            print(file=sys.stderr, flush=True)

        sites: List[SiteKey] = sorted(known_sites_by_index)

        if save_extra_positions is not None:
            _write_extra_positions(
                save_extra_positions,
                sites=sites,
                seeded_site_keys=seeded_site_keys,
                contig_names=contig_names,
            )

        out_of_known_alleles_site_proportion = (
            float(len(known_sites_with_out_of_set_bases)) / float(known_sites_count)
            if known_sites_count > 0
            else 0.0
        )

        num_samples = len(sample_ids)
        num_variants = len(sites)

        site_index = {site: i for i, site in enumerate(sites)}
        max_alleles = max((len(known_sites_by_index[site]) for site in sites), default=1)
        allele_index_by_site: Dict[PositionKey, Dict[str, int]] = {
            site: {allele: idx for idx, allele in enumerate(known_sites_by_index[site])}
            for site in sites
        }

        chunk_v = max(1, min(num_variants, 4096))
        chunk_s = max(1, min(num_samples, 1024))

        variant_contig = np.array([site[0] for site in sites], dtype=np.int32)
        variant_position = np.array([site[1] for site in sites], dtype=np.int32)
        variant_allele = np.full((num_variants, max_alleles), "", dtype=object)
        for idx, site in enumerate(sites):
            alleles = known_sites_by_index[site]
            variant_allele[idx, : len(alleles)] = alleles

        def _float_attr(value: float | None) -> float:
            return float("nan") if value is None else value

        # Write the zarr store directly so call_genotype can be streamed
        # in sample-batches without ever mapping the full matrix into VM.
        root = zarr.open_group(output_zarr_path, mode="w")
        root.attrs.update({
            "vcf_zarr_version": "0.4",
            "source": "bam2zarr",
            "known_sites_count": known_sites_count,
            "out_of_known_alleles_site_proportion": out_of_known_alleles_site_proportion,
            "tree_sequence_comparable_site_count": tree_sequence_reference_summary.comparable_site_count,
            "tree_sequence_ancestral_ref_mismatch_site_count": tree_sequence_reference_summary.ancestral_ref_mismatch_site_count,
            "tree_sequence_ancestral_ref_match_proportion": _float_attr(tree_sequence_reference_summary.ancestral_ref_match_proportion),
            "tree_sequence_ancestral_ref_mismatch_proportion": _float_attr(tree_sequence_reference_summary.ancestral_ref_mismatch_proportion),
        })

        def _zwrite(name: str, data: np.ndarray, dims: List[str], **kw: object) -> None:
            if np.asarray(data).dtype.kind == "O":
                kw.setdefault("object_codec", numcodecs.VLenUTF8())
            arr = root.array(name, data, **kw)
            arr.attrs["_ARRAY_DIMENSIONS"] = dims

        _zwrite("contig_id", np.array(contig_names, dtype=object), ["contigs"])
        _zwrite("sample_id", np.array(sample_ids, dtype=object), ["samples"])
        # tskit MetadataSchema.decode_row expects bytes-like payloads.
        individuals_metadata_array = root.array(
            "individuals_metadata",
            np.array(individuals_metadata, dtype=object),
            object_codec=numcodecs.VLenBytes(),
        )
        individuals_metadata_array.attrs["_ARRAY_DIMENSIONS"] = ["samples"]
        _zwrite("variant_contig", variant_contig, ["variants"], chunks=(chunk_v,))
        _zwrite("variant_position", variant_position, ["variants"], chunks=(chunk_v,))
        _zwrite("variant_allele", variant_allele, ["variants", "alleles"],
                chunks=(chunk_v, max_alleles))

        # Create call_genotype with fill_value=-1 without writing any data yet.
        # Unwritten chunks are served as fill_value by zarr, so only the batches
        # we explicitly write consume RAM (num_variants * chunk_s bytes each).
        cg = root.create_dataset(
            "call_genotype",
            shape=(num_variants, num_samples, 1),
            chunks=(chunk_v, chunk_s, 1),
            dtype=np.int8,
            fill_value=-1,
        )
        cg.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

        bam.close()
        bam = pysam.AlignmentFile(bam_path, "rb")
        sample_seen_pass2: Dict[str, int] = defaultdict(int)
        sample_idx = 0
        batch_start = 0
        last_progress_update = 0.0

        # One batch at a time: only num_variants * chunk_s bytes in RAM.
        batch = np.full((num_variants, chunk_s, 1), -1, dtype=np.int8)

        if total_reads is not None:
            _print_bam_progress(
                0,
                total_reads,
                kept_reads=0,
                label="Reading BAM (pass 2/2)",
            )

        for processed_reads, read in enumerate(bam.fetch(until_eof=True), start=1):
            if read.is_unmapped or read.mapping_quality < min_mapq:
                if total_reads is not None:
                    now = time.monotonic()
                    if processed_reads == total_reads or now - last_progress_update >= 0.2:
                        _print_bam_progress(
                            processed_reads,
                            total_reads,
                            kept_reads=sample_idx,
                            label="Reading BAM (pass 2/2)",
                        )
                        last_progress_update = now
                continue

            if not include_secondary and (read.is_secondary or read.is_supplementary):
                if total_reads is not None:
                    now = time.monotonic()
                    if processed_reads == total_reads or now - last_progress_update >= 0.2:
                        _print_bam_progress(
                            processed_reads,
                            total_reads,
                            kept_reads=sample_idx,
                            label="Reading BAM (pass 2/2)",
                        )
                        last_progress_update = now
                continue

            base_name = read.query_name or f"read_{sample_idx}"
            _unique_sample_name(base_name, sample_seen_pass2)
            query_sequence = read.query_sequence or ""

            pos_in_batch = sample_idx - batch_start
            for query_pos, ref_pos in read.get_aligned_pairs(matches_only=True):
                if query_pos is None or ref_pos is None:
                    continue
                if ref_pos < 0 or ref_pos >= bam.lengths[read.reference_id]:
                    continue

                read_base = query_sequence[query_pos].upper()
                if read_base not in {"A", "C", "G", "T"}:
                    continue

                contig_name = read.reference_name
                if contig_name is None:
                    continue

                contig_index = contig_to_index.get(contig_name)
                if contig_index is None:
                    continue

                pos1 = ref_pos + 1
                key = (contig_index, pos1)
                allele_index = allele_index_by_site.get(key, {}).get(read_base)
                if allele_index is not None:
                    batch[site_index[key], pos_in_batch, 0] = allele_index

            sample_idx += 1
            # Flush a complete batch column to zarr when full.
            if sample_idx - batch_start == chunk_s:
                cg[:, batch_start:batch_start + chunk_s, :] = batch
                batch[:] = -1
                batch_start += chunk_s


            if total_reads is not None:
                now = time.monotonic()
                if processed_reads == total_reads or now - last_progress_update >= 0.2:
                    _print_bam_progress(
                        processed_reads,
                        total_reads,
                        kept_reads=sample_idx,
                        label="Reading BAM (pass 2/2)",
                    )
                    last_progress_update = now

        if total_reads is not None:
            print(file=sys.stderr, flush=True)

        # Flush the final partial batch (fewer than chunk_s samples).
        if sample_idx > batch_start:
            remaining = sample_idx - batch_start
            cg[:, batch_start:batch_start + remaining, :] = batch[:, :remaining, :]

        zarr.consolidate_metadata(output_zarr_path)

        if not Path(output_zarr_path).exists():
            raise FileNotFoundError(
                f"Expected output Zarr store was not created: {output_zarr_path}"
            )

        return ConversionSummary(
            sample_count=num_samples,
            variant_count=num_variants,
            contig_count=len(contig_names),
            known_sites_count=known_sites_count,
            out_of_known_alleles_site_proportion=out_of_known_alleles_site_proportion,
            tree_sequence_comparable_site_count=tree_sequence_reference_summary.comparable_site_count,
            tree_sequence_ancestral_ref_mismatch_site_count=tree_sequence_reference_summary.ancestral_ref_mismatch_site_count,
            tree_sequence_ancestral_ref_match_proportion=tree_sequence_reference_summary.ancestral_ref_match_proportion,
            tree_sequence_ancestral_ref_mismatch_proportion=tree_sequence_reference_summary.ancestral_ref_mismatch_proportion,
        )
    finally:
        if 'bam' in locals() and bam is not None:
            bam.close()
        if reference_fasta is not None:
            reference_fasta.close()
