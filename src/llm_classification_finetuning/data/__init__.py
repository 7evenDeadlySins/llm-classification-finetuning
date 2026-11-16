"""Raw competition data acquisition."""

from .downloader import CompetitionDataDownloader, DownloadResult
from .external import (
	EXTERNAL_SOURCES,
	ExternalSource,
	ExternalSourceDownloader,
	file_sha256,
)

__all__ = (
	"CompetitionDataDownloader",
	"DownloadResult",
	"EXTERNAL_SOURCES",
	"ExternalSource",
	"ExternalSourceDownloader",
	"file_sha256",
)
