"""Convert official human preference records into aligned hard-label battles."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import tokenize
from typing import Any

from ..data.external import ExternalSource


TARGET_COLUMNS = ("winner_model_a", "winner_model_b", "winner_tie")


def canonical_json(value: Any) -> str:
	return json.dumps(
		value,
		ensure_ascii=False,
		sort_keys=True,
		separators=(",", ":"),
		allow_nan=False,
	)


class ConversationExclusion(ValueError):
	"""A source row cannot be represented under the adopted text-only policy."""

	def __init__(self, reason: str, detail: str) -> None:
		super().__init__(detail)
		self.reason = reason


class ContainerRepresentationDecoder:
	"""Read the expert release's NumPy repr without executing source expressions.

	NumPy's outer list representation omits commas between objects. Only literal
	containers and array([...], dtype=object) wrappers are accepted. String tokens
	are decoded individually, so punctuation inside user text is never rewritten.
	"""

	def decode(self, value: Any) -> list:
		if isinstance(value, list):
			return value
		if not isinstance(value, str):
			raise ValueError(
				"Conversation must be a list or a container representation."
			)
		try:
			parsed = json.loads(value)
		except json.JSONDecodeError:
			ignored = {
				tokenize.NL,
				tokenize.NEWLINE,
				tokenize.INDENT,
				tokenize.DEDENT,
				tokenize.ENDMARKER,
			}
			self.tokens = [
				t.string
				for t in tokenize.generate_tokens(io.StringIO(value).readline)
				if t.type not in ignored
			]
			self.position = 0
			parsed = self._value()
			if self.position != len(self.tokens):
				raise ValueError("Unexpected trailing conversation representation.")
		if not isinstance(parsed, list):
			raise ValueError("Conversation representation is not a list.")
		return parsed

	def _take(self, expected: str | None = None) -> str:
		if self.position >= len(self.tokens):
			raise ValueError("Incomplete container representation.")
		value = self.tokens[self.position]
		self.position += 1
		if expected is not None and value != expected:
			raise ValueError(f"Expected container token {expected!r}.")
		return value

	def _value(self) -> Any:
		token = self._take()
		if token == "array":
			self._take("(")
			value = self._value()
			self._take(",")
			self._take("dtype")
			self._take("=")
			dtype = self._take()
			if dtype != "object" and not (dtype == "float64" and value == []):
				raise ValueError(
					"Only object arrays or empty float64 arrays are supported."
				)
			self._take(")")
			if not isinstance(value, list):
				raise ValueError("Only list-valued object arrays are supported.")
			return value
		if token in ("[", "{"):
			closing = "]" if token == "[" else "}"
			items = []
			mapping = {}
			while (
				self.position < len(self.tokens)
				and self.tokens[self.position] != closing
			):
				value = self._value()
				if token == "{":
					if not isinstance(value, str) or value in mapping:
						raise ValueError("Object keys must be unique strings.")
					self._take(":")
					mapping[value] = self._value()
				else:
					items.append(value)
				if (
					self.position < len(self.tokens)
					and self.tokens[self.position] == ","
				):
					self._take(",")
				elif (
					token == "{"
					and self.position < len(self.tokens)
					and self.tokens[self.position] != closing
				):
					raise ValueError("Missing object separator.")
			self._take(closing)
			return items if token == "[" else mapping
		if token == "...":
			raise ValueError("Source representation elides conversation content.")
		value = ast.literal_eval(token)
		if not isinstance(value, (str, int, float, bool, type(None))):
			raise ValueError("Unsupported container literal.")
		return value


class ExternalConversationConverter:
	"""Preserve text, complete context, and vote provenance for one pinned source."""

	WINNERS = {"model_a": 0, "model_b": 1, "tie": 2, "both_bad": 2, "tie (bothbad)": 2}

	def __init__(self, source: ExternalSource) -> None:
		self.source = source
		self.decoder = ContainerRepresentationDecoder()

	@staticmethod
	def _text(content: Any, *, user: bool = False) -> str:
		if isinstance(content, str):
			return content
		if content is None and not user:
			return ""
		if not isinstance(content, list):
			raise ConversationExclusion(
				"excluded_structure", "Message content is not text or a content list."
			)
		texts = []
		for part in content:
			if not isinstance(part, dict):
				raise ConversationExclusion(
					"excluded_structure", "Invalid message content part."
				)
			if part.get("type") != "text" or any(
				v not in (None, "")
				for k, v in part.items()
				if k not in ("type", "text", "mimeType")
			):
				raise ConversationExclusion(
					"excluded_multimodal",
					"A required message contains non-text content.",
				)
			text = part.get("text")
			if not isinstance(text, str):
				raise ConversationExclusion(
					"excluded_structure", "Text content is not a string."
				)
			texts.append(text)
		return "".join(texts)

	def _messages(self, messages: list) -> list[dict[str, str]]:
		if not messages or len(messages) % 2:
			raise ConversationExclusion(
				"excluded_structure", "Expected complete user/assistant pairs."
			)
		result = []
		for i, message in enumerate(messages):
			role = "assistant" if i % 2 else "user"
			if not isinstance(message, dict) or message.get("role") != role:
				raise ConversationExclusion(
					"excluded_structure", f"Expected {role} at position {i}."
				)
			result.append(
				{
					"role": role,
					"content": self._text(message.get("content"), user=role == "user"),
				}
			)
		return result

	def _arena(self, row: dict) -> tuple[list, list]:
		a, b = (
			self._messages(self.decoder.decode(row[key]))
			for key in ("conversation_a", "conversation_b")
		)
		if len(a) != len(b):
			raise ConversationExclusion(
				"excluded_structure", "Current branch turn counts differ."
			)
		full = self.decoder.decode(row["full_conversation"])
		ka, kb = (
			("model_side_a", "model_side_b")
			if self.source.name == "arena_human"
			else ("model_a", "model_b")
		)
		if not full or any(
			not isinstance(turn, dict) or not {"user", ka, kb}.issubset(turn)
			for turn in full
		):
			raise ConversationExclusion(
				"excluded_structure", "Missing complete Arena history fields."
			)
		fa = [message for turn in full for message in (turn["user"], turn[ka])]
		fb = [message for turn in full for message in (turn["user"], turn[kb])]
		matches = []
		for start in range(0, len(fa) - len(a) + 1, 2):
			try:
				if (
					self._messages(fa[start : start + len(a)]) == a
					and self._messages(fb[start : start + len(b)]) == b
				):
					matches.append(start + len(a))
			except ConversationExclusion:
				continue
		if len(matches) != 1:
			raise ConversationExclusion(
				"excluded_structure",
				"Current evaluation cannot be uniquely aligned with full history.",
			)
		return self._messages(fa[: matches[0]]), self._messages(fb[: matches[0]])

	def convert(self, row: dict, index: int, source_split: str) -> dict:
		try:
			if self.source.name == "helpsteer3":
				context = row["context"]
				if (
					not isinstance(context, list)
					or not context
					or len(context) % 2 != 1
				):
					raise ConversationExclusion(
						"excluded_structure",
						"HelpSteer context must end with a user message.",
					)
				a = self._messages(
					[*context, {"role": "assistant", "content": row["response1"]}]
				)
				b = self._messages(
					[*context, {"role": "assistant", "content": row["response2"]}]
				)
				score = row["overall_preference"]
				if type(score) is not int or score not in range(-3, 4):
					raise ValueError("Invalid overall_preference.")
				label = 0 if score < 0 else 1 if score > 0 else 2
				source_id = hashlib.sha256(
					canonical_json(row).encode("utf-8")
				).hexdigest()
				vote_id = f"helpsteer3:{source_id}"
				raw_label = str(score)
			elif self.source.name == "lmsys":
				a, b = (
					self._messages(row[key])
					for key in ("conversation_a", "conversation_b")
				)
				label = self.WINNERS[row["winner"]]
				source_id = row["question_id"]
				vote_id = "lmsys:" + canonical_json(
					[source_id, row["judge"], row["tstamp"]]
				)
				raw_label = row["winner"]
			else:
				a, b = self._arena(row)
				label = self.WINNERS[row["winner"]]
				source_id = row["id"]
				vote_id = f"arena:{source_id}"
				raw_label = row["winner"]
			if not isinstance(source_id, str) or not source_id:
				raise ValueError("Missing source vote identifier.")
			prompts = [m["content"] for m in a[::2]]
			if prompts != [m["content"] for m in b[::2]]:
				raise ConversationExclusion(
					"excluded_structure", "Complete A/B user prompts differ."
				)
			answers_a, answers_b = (
				[m["content"] for m in branch[1::2]] for branch in (a, b)
			)
			original_label = label
			if answers_a == answers_b or (not answers_a[-1] and not answers_b[-1]):
				label = 2
			prompt = canonical_json(prompts)
			return {
				"id": self.source.id_segment * 2**32 + index,
				"model_a": row.get("model_a", "response1"),
				"model_b": row.get("model_b", "response2"),
				"prompt": prompt,
				"response_a": canonical_json(answers_a),
				"response_b": canonical_json(answers_b),
				"label": label,
				**{name: int(label == i) for i, name in enumerate(TARGET_COLUMNS)},
				"group_id": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
				"split": "train",
				"is_swapped": False,
				"source": self.source.name,
				"source_id": source_id,
				"source_row_index": index,
				"source_split": source_split,
				"source_winner": raw_label,
				"vote_id": vote_id,
				"original_label": original_label,
				"tie_corrected": original_label != label,
				"language": row.get("language"),
				"domain": row.get("domain"),
			}
		except ConversationExclusion:
			raise
		except (
			ValueError,
			TypeError,
			KeyError,
			SyntaxError,
			tokenize.TokenError,
			RecursionError,
		) as error:
			raise ConversationExclusion(
				"excluded_structure", f"{type(error).__name__}: {error}"
			) from error
