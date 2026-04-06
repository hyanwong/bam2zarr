def test_import() -> None:
    from bam2zarr import convert_bam_to_vcfzarr

    assert callable(convert_bam_to_vcfzarr)
