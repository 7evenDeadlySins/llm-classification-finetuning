"""Acquire immutable official preference releases and verify every input file."""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from ..errors import DataPreparationError


def file_sha256(path: Path) -> str:
	"""Hash a file without loading it into memory."""
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ExternalSource:
	name: str
	repository: str
	revision: str
	id_segment: int
	files: tuple[str, ...]
	expected_rows: int


EXTERNAL_SOURCES = (
	ExternalSource(
		"lmsys",
		"lmsys/chatbot_arena_conversations",
		"1b6335d42a1d2c7e34870c905d03ab964f7f2bd8",
		1,
		("data/train-00000-of-00001-cced8514c7ed782a.parquet",),
		33000,
	),
	ExternalSource(
		"arena_human",
		"lmarena-ai/arena-human-preference-140k",
		"6322995ab34d7c2693e3f47dd13fa5caa0789a74",
		2,
		tuple(f"data/train-{i:05d}-of-00007.parquet" for i in range(7)),
		135634,
	),
	ExternalSource(
		"arena_expert",
		"lmarena-ai/arena-expert-5k",
		"171f77047d8d153c41b3f2c11bc910273f62f183",
		3,
		("data/train-00000-of-00001.parquet",),
		5128,
	),
)


class ExternalSourceDownloader:
	"""Retain source files plus a checksummed lock manifest below the raw directory."""

	def __init__(self, raw_dir: Path) -> None:
		self.root = raw_dir.resolve() / "external"
		self.api = HfApi(endpoint="https://huggingface.co")

	def ensure_available(self) -> dict:
		self.root.mkdir(parents=True, exist_ok=True)
		lock = self.root / "sources.json"
		if lock.is_file():
			manifest = json.loads(lock.read_text(encoding="utf-8"))
			if manifest.get("sources") != [
				json.loads(json.dumps(asdict(s))) for s in EXTERNAL_SOURCES
			]:
				raise DataPreparationError(
					"External source lock does not match pinned releases."
				)
			for item in manifest["files"]:
				path = self.root / item["local_path"]
				if (
					not path.is_file()
					or path.stat().st_size != item["bytes"]
					or file_sha256(path) != item["sha256"]
				):
					raise DataPreparationError(
						f"Locked external source is missing or corrupt: {path}"
					)
			print("Reusing all checksummed external source files.", flush=True)
			return manifest
		jobs = []
		for source in EXTERNAL_SOURCES:
			info = self.api.dataset_info(
				source.repository, revision=source.revision, files_metadata=True
			)
			for filename in source.files:
				metadata = next(
					item for item in info.siblings if item.rfilename == filename
				)
				if metadata.lfs is None:
					raise DataPreparationError(
						f"Missing official LFS checksum: {source.name}/{filename}"
					)
				jobs.append((source, filename, metadata.size, metadata.lfs.sha256))
		with ThreadPoolExecutor(max_workers=3) as pool:
			files = list(pool.map(self._download, jobs))
		manifest = {
			"sources": [asdict(source) for source in EXTERNAL_SOURCES],
			"files": files,
		}
		staged = lock.with_suffix(".json.tmp")
		staged.write_text(
			json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
		)
		staged.replace(lock)
		return manifest

	def _download(self, job: tuple) -> dict:
		source, filename, size, checksum = job
		relative = Path(source.name) / source.revision / filename
		destination = self.root / relative
		destination.parent.mkdir(parents=True, exist_ok=True)
		if (
			not destination.is_file()
			or destination.stat().st_size != size
			or file_sha256(destination) != checksum
		):
			print(f"Downloading {source.name}: {filename}", flush=True)
			path = Path(
				hf_hub_download(
					source.repository,
					filename,
					repo_type="dataset",
					revision=source.revision,
					cache_dir=self.root / "hub",
					endpoint="https://huggingface.co",
				)
			)
			if path.stat().st_size != size or file_sha256(path) != checksum:
				raise DataPreparationError(
					f"Official checksum mismatch: {source.name}/{filename}"
				)
			working = destination.with_suffix(destination.suffix + ".tmp")
			shutil.copyfile(path, working)
			working.replace(destination)
		print(f"Verified {source.name}: {filename} ({size:,} bytes)", flush=True)
		return {
			"source": source.name,
			"filename": filename,
			"local_path": relative.as_posix(),
			"bytes": size,
			"sha256": checksum,
		}
