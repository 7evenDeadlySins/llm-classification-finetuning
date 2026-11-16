"""Merge pinned real human preferences while preserving the saved Kaggle holdout."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from llm_classification_finetuning.cli_config import add_config_argument
from llm_classification_finetuning.config import AppConfig
from llm_classification_finetuning.errors import DataPreparationError
from llm_classification_finetuning.external_pipeline import MergeExternalDataPipeline


def main(arguments: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	add_config_argument(parser)
	parser.add_argument(
		"--baseline-path",
		type=Path,
		default=Path("data/processed/lmsys33k/train_validation.parquet"),
	)
	parser.add_argument(
		"--output-dir", type=Path, default=Path("data/processed/external_merged")
	)
	options = parser.parse_args(arguments)
	try:
		output = MergeExternalDataPipeline(
			AppConfig.load(options.config),
			baseline_path=options.baseline_path,
			output_dir=options.output_dir,
		).run()
		print(f"Merge summary: {output / 'summary.json'}")
		return 0
	except (DataPreparationError, OSError) as error:
		print(f"External data merge failed: {error}", flush=True)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
