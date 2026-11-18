"""Generate audited train-partition or full-data prioritized tie duplicate supplements."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from llm_classification_finetuning.cli_config import add_config_argument
from llm_classification_finetuning.config import AppConfig
from llm_classification_finetuning.errors import DataPreparationError
from llm_classification_finetuning.preprocess import TieDuplicateGenerator


def main(arguments: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	add_config_argument(parser)
	parser.add_argument("--scope", choices=("train", "build", "all"), default="all")
	parser.add_argument(
		"--target-count",
		type=int,
		default=TieDuplicateGenerator.DEFAULT_TARGET_COUNT,
	)
	parsed_arguments = parser.parse_args(arguments)
	try:
		config = AppConfig.load(parsed_arguments.config)
		scopes = (
			("train", "build")
			if parsed_arguments.scope == "all"
			else (parsed_arguments.scope,)
		)
		enabled = [
			scope
			for scope in scopes
			if getattr(config.data, f"{scope}_supplement_path") is not None
		]
		if not enabled or (
			parsed_arguments.scope != "all" and len(enabled) != len(scopes)
		):
			raise DataPreparationError(
				"Configure the requested data.train_supplement_path or data.build_supplement_path."
			)
		for scope in enabled:
			output = TieDuplicateGenerator(
				config,
				scope=scope,
				target_count=parsed_arguments.target_count,
			).run()
			print(f"Generation summary ({scope}): {output / 'summary.json'}")
		return 0
	except (DataPreparationError, OSError) as error:
		print(f"Tie duplicate generation failed: {error}")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
