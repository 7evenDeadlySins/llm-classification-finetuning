"""Orchestrate prebuilt data availability and raw-data preparation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig
from .data import CompetitionDataDownloader, DownloadResult
from .errors import DataPreparationError
from .models.common import CacheStorage, TrainingInputLoader
from .preprocess import SplitPreparationResult, SplitPreprocessor


@dataclass(frozen=True, slots=True)
class PrepareDataResult:
	download: DownloadResult
	splits: SplitPreparationResult


class PrepareDataPipeline:
	"""Run the complete raw-data-to-training-artifact workflow."""

	def __init__(self, config: AppConfig) -> None:
		self._config = config
		self._downloader = CompetitionDataDownloader(config.data)
		self._preprocessor = SplitPreprocessor(
			config.validation, cache_dir=config.cache_dir
		)

	def run(
		self, on_download: Callable[[DownloadResult], None] | None = None
	) -> PrepareDataResult:
		"""Ensure raw inputs and preprocess them."""
		download_result = self._downloader.ensure_available()
		if on_download is not None:
			on_download(download_result)
		split_result = self._preprocessor.prepare(
			train_path=download_result.file_paths["train.csv"],
			output_path=self._config.data.processed_path,
		)
		return PrepareDataResult(download=download_result, splits=split_result)


class EnsureProcessedDataPipeline:
	"""Reuse local processed data or fill missing files from a project mirror."""

	def __init__(
		self,
		config: AppConfig,
		*,
		project_root: Path,
		source_root: Path | None = None,
		include_build_supplement: bool = False,
	) -> None:
		self._storages = [CacheStorage(config.data.processed_path)]
		self._token_stats_storages: tuple[CacheStorage, ...] = ()
		if config.data.external_rank_path is not None:
			self._storages.append(CacheStorage(config.data.external_rank_path))
			self._token_stats_storages = tuple(
				CacheStorage(config.data.processed_path.with_name(name))
				for name in TrainingInputLoader.TOKEN_STATS_FILES
			)
		supplement_path = (
			config.data.build_supplement_path
			if include_build_supplement
			else config.data.train_supplement_path
		)
		if supplement_path is not None:
			self._storages.append(CacheStorage(supplement_path))
		self._project_root = project_root.resolve()
		# Resolve the source only after a local miss, avoiding access to Drive on hits.
		self._source_root = source_root

	def run(
		self, on_copy: Callable[[Path, Path], None] | None = None
	) -> tuple[Path, ...]:
		"""Ensure required data and pick up token statistics when using a mirror."""
		copied = tuple(
			path
			for storage in self._storages
			for path in self._ensure(storage, on_copy)
		)
		if not copied:
			return copied
		return copied + tuple(
			path
			for storage in self._token_stats_storages
			for path in self._ensure(storage, on_copy, optional=True)
		)

	def _ensure(
		self,
		storage: CacheStorage,
		on_copy: Callable[[Path, Path], None] | None,
		*,
		optional: bool = False,
	) -> tuple[Path, ...]:
		source_path: Path | None = None
		try:
			missing = storage.missing_components()
			if not missing:
				return ()
			missing_paths = ", ".join(str(path) for path in missing)
			if self._source_root is None:
				raise DataPreparationError(
					f"Missing or wrong-sized processed data: {missing_paths}. "
					f"Expected {storage.path} or {storage.manifest_path}. "
					"No --data-source-root configured."
				)
			try:
				relative_path = storage.path.relative_to(self._project_root)
			except ValueError as error:
				if optional:
					return ()
				raise DataPreparationError(
					f"Missing or wrong-sized processed data: {missing_paths}. "
					f"Cannot map {storage.path} to the source root: "
					f"it is outside the project root {self._project_root}."
				) from error
			source_root = self._source_root.expanduser()
			if not source_root.is_absolute():
				source_root = self._project_root / source_root
			source_path = source_root / relative_path
			source = CacheStorage(source_path)
			if optional and source.missing_components():
				return ()
			return storage.copy_missing_from(source, on_copy=on_copy)
		except OSError as error:
			source_detail = f" from {source_path}" if source_path is not None else ""
			raise DataPreparationError(
				f"Could not prepare processed data {storage.path}{source_detail}: {error}"
			) from error
