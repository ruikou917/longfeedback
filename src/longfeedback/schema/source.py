"""Dataset identity, licensing, and derivation provenance."""

from __future__ import annotations

from pydantic import Field, HttpUrl

from longfeedback.schema.base import FrozenRecord


class SourceManifest(FrozenRecord):
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_license: str = Field(min_length=1)
    source_url: HttpUrl
    derivative_license: str = Field(min_length=1)
    redistribute_raw_text: bool = False
    required_attribution: bool = True
    pii_filter_version: str = Field(min_length=1)
    labeler_version: str = Field(min_length=1)
    source_checksum: str = Field(min_length=1)
