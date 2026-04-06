"""CLI for BAM to VCF Zarr conversion."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from .converter import convert_bam_to_vcfzarr


def _parse_tree_sequence_entry(value: str) -> tuple[str, str]:
    contig, separator, path = value.partition("=")
    if separator != "=" or not contig or not path:
        raise argparse.ArgumentTypeError(
            "Tree-sequence mapping must use CONTIG=PATH format"
        )
    return contig, path


def _load_tree_sequences(entries: list[str]) -> dict[str, object]:
    if not entries:
        return {}

    try:
        tskit = importlib.import_module("tskit")
    except ImportError as exc:
        raise SystemExit(
            "The --tree-sequence option requires tskit. Install it and retry."
        ) from exc

    tree_sequences: dict[str, object] = {}
    for entry in entries:
        contig, path = _parse_tree_sequence_entry(entry)
        if contig in tree_sequences:
            raise SystemExit(f"Duplicate --tree-sequence contig name: {contig}")
        if not Path(path).exists():
            raise SystemExit(f"Tree-sequence path does not exist: {path}")
        tree_sequences[contig] = tskit.load(path)

    return tree_sequences


def _default_output_path(bam_path: str) -> str:
    bam = Path(bam_path)
    if bam.suffix:
        return str(bam.with_suffix(".vcz"))
    return f"{bam_path}.vcz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bam2zarr",
        description="Convert BAM reads into a VCF Zarr-compatible dataset.",
    )
    parser.add_argument("bam", help="Input BAM file path")
    parser.add_argument(
        "reference_fasta",
        help="Reference FASTA used to discover fragment mismatches and set REF alleles",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output Zarr store path (default: BAM path with .vcz extension)",
    )
    parser.add_argument(
        "--known-sites",
        default=None,
        help="Known-sites source path (CSV/TSV, VCF/BCF, or VCF Zarr)",
    )
    parser.add_argument(
        "--tree-sequence",
        action="append",
        default=[],
        metavar="CONTIG=PATH",
        help="Map a contig to a tree-sequence file; repeat per contig",
    )
    parser.add_argument(
        "--tree-sequence-position-base",
        type=int,
        choices=(0, 1),
        default=1,
        help="Interpret tree-sequence positions as 0-based or 1-based (default: 1)",
    )
    parser.add_argument(
        "--save-extra-positions",
        default=None,
        help="Write output variant positions absent from the known-sites input as contig<TAB>position",
    )
    parser.add_argument(
        "--min-mapq",
        type=int,
        default=0,
        help="Minimum mapping quality to keep a read (default: 0)",
    )
    parser.add_argument(
        "--include-secondary",
        action="store_true",
        help="Include secondary/supplementary alignments",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress-bar and progress diagnostics on stderr",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tree_sequences = _load_tree_sequences(args.tree_sequence)
    output_path = args.output if args.output is not None else _default_output_path(args.bam)

    if tree_sequences and args.known_sites is not None:
        raise SystemExit(
            "Provide either --known-sites or --tree-sequence entries, not both"
        )

    known_sites_input: object | None
    if tree_sequences:
        known_sites_input = tree_sequences
    else:
        known_sites_input = args.known_sites

    convert_bam_to_vcfzarr(
        bam_path=args.bam,
        output_zarr_path=output_path,
        known_sites=known_sites_input,
        reference_fasta_path=args.reference_fasta,
        save_extra_positions=args.save_extra_positions,
        tree_sequence_position_base=args.tree_sequence_position_base,
        min_mapq=args.min_mapq,
        include_secondary=args.include_secondary,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
