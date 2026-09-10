"""Tests for the collection -> SFT dataset conversion (KISS spec 03, T1-T15)."""

import json

import pytest
from slam_core.collections.base import CollectionInfo, EvalCase, EvalCaseCollection
from slam_core.collections.text_generation import TextGenerationInput
from slam_core.scorers.merge_quality_scorer import safe_parse_prediction

from slam_train.data.sft_dataset import (
    build_records,
    materialize_collection,
    serialize_completion,
    split_cases,
    split_indices,
)


class FakeDictCollection(EvalCaseCollection):
    """Mimics MergeQuality: y_true is a dict, no system prompt."""

    def __init__(self, name: str, items: list[dict] | None = None) -> None:
        super().__init__(name)
        self.items = items or [
            {"a": 1, "b": {"c": "x"}},
            {"a": 2, "b": {"c": "y"}},
        ]

    def _load(self) -> CollectionInfo:
        return CollectionInfo(
            collection=iter(
                [
                    EvalCase(
                        x=TextGenerationInput(
                            system_prompt=None, user_prompt=f"prompt {i}"
                        ),
                        y_true=item,
                    )
                    for i, item in enumerate(self.items)
                ]
            ),
            collection_len=len(self.items),
        )

    def __next__(self) -> EvalCase:
        return next(self.collection)  # type: ignore


class FakeStrCollection(EvalCaseCollection):
    """Mimics BigBenchHard: y_true is a str, no system prompt."""

    def __init__(self, name: str, items: list[str] | None = None) -> None:
        super().__init__(name)
        self.items = items or ["answer 1", "answer 2"]

    def _load(self) -> CollectionInfo:
        return CollectionInfo(
            collection=iter(
                [
                    EvalCase(
                        x=TextGenerationInput(
                            system_prompt=None, user_prompt=f"prompt {i}"
                        ),
                        y_true=item,
                    )
                    for i, item in enumerate(self.items)
                ]
            ),
            collection_len=len(self.items),
        )

    def __next__(self) -> EvalCase:
        return next(self.collection)  # type: ignore


class FakeSystemPromptCollection(EvalCaseCollection):
    """Emits a non-None system prompt."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _load(self) -> CollectionInfo:
        data = [
            EvalCase(
                x=TextGenerationInput(system_prompt="be nice", user_prompt="hi"),
                y_true="ok",
            )
        ]
        return CollectionInfo(collection=iter(data), collection_len=len(data))

    def __next__(self) -> EvalCase:
        return next(self.collection)  # type: ignore


class TestMaterializeCollection:
    def test_drains_the_iterator(self):
        collection = FakeDictCollection("fake", items=[{"a": i} for i in range(5)])
        cases = materialize_collection(collection)
        assert len(cases) == 5


class TestBuildRecords:
    def test_prompt_is_user_prompt_verbatim(self):
        collection = FakeDictCollection("fake")
        cases = materialize_collection(collection)
        records = build_records(cases)
        assert records[0]["prompt"][-1]["content"] == "prompt 0"

    def test_records_use_prompt_completion_keys(self):
        cases = materialize_collection(FakeStrCollection("fake"))
        records = build_records(cases)
        assert set(records[0].keys()) == {"prompt", "completion"}

    def test_dict_y_true_serialized_as_json(self):
        cases = materialize_collection(FakeDictCollection("fake"))
        records = build_records(cases)
        expected = json.dumps(cases[0]["y_true"], ensure_ascii=False, indent=2)
        assert records[0]["completion"] == [{"role": "assistant", "content": expected}]

    def test_round_trip_through_safe_parse_prediction(self):
        cases = materialize_collection(FakeDictCollection("fake"))
        records = build_records(cases)
        parsed = safe_parse_prediction(records[0]["completion"][0]["content"])
        assert parsed == cases[0]["y_true"]

    def test_str_y_true_used_verbatim(self):
        cases = materialize_collection(FakeStrCollection("fake"))
        records = build_records(cases)
        assert records[0]["completion"] == [
            {"role": "assistant", "content": "answer 1"}
        ]

    def test_no_system_message_when_system_prompt_is_none(self):
        cases = materialize_collection(FakeDictCollection("fake"))
        records = build_records(cases)
        roles = [m["role"] for m in records[0]["prompt"]]
        assert roles == ["user"]

    def test_system_prompt_preserved_when_set(self):
        cases = materialize_collection(FakeSystemPromptCollection("fake"))
        records = build_records(cases)
        assert records[0]["prompt"][0] == {"role": "system", "content": "be nice"}


class TestSerializeCompletion:
    def test_str_passthrough(self):
        assert serialize_completion("abc") == "abc"

    def test_dict_json_dumps(self):
        assert serialize_completion({"a": 1}) == json.dumps({"a": 1}, indent=2)


class TestSplitCases:
    def _make_cases(self, n: int) -> list[EvalCase]:
        return [
            EvalCase(
                x=TextGenerationInput(system_prompt=None, user_prompt=f"p{i}"),
                y_true=str(i),
            )
            for i in range(n)
        ]

    def test_ratio_0_8_over_100_gives_80_20(self):
        train, eval_ = split_cases(self._make_cases(100), 0.8, seed=42)
        assert len(train) == 80
        assert len(eval_) == 20

    def test_subsets_disjoint_and_complete(self):
        cases = self._make_cases(50)
        train, eval_ = split_cases(cases, 0.8, seed=1)
        train_prompts = {c["x"]["user_prompt"] for c in train}
        eval_prompts = {c["x"]["user_prompt"] for c in eval_}
        assert not train_prompts & eval_prompts
        assert train_prompts | eval_prompts == {f"p{i}" for i in range(50)}

    def test_split_is_not_raw_order(self):
        cases = self._make_cases(100)
        train, _ = split_cases(cases, 0.8, seed=7)
        train_prompts = [c["x"]["user_prompt"] for c in train]
        assert train_prompts != [f"p{i}" for i in range(80)]

    def test_same_seed_same_split(self):
        cases = self._make_cases(30)
        train1, eval1 = split_cases(cases, 0.8, seed=3)
        train2, eval2 = split_cases(cases, 0.8, seed=3)
        assert train1 == train2
        assert eval1 == eval2

    def test_different_seed_different_split(self):
        cases = self._make_cases(30)
        train1, eval1 = split_cases(cases, 0.8, seed=3)
        train2, eval2 = split_cases(cases, 0.8, seed=4)
        assert train1 != train2
        assert len(train1) == len(train2) == 24
        assert len(eval1) == len(eval2) == 6

    def test_ratio_1_0_all_train(self):
        cases = self._make_cases(10)
        train, eval_ = split_cases(cases, 1.0, seed=0)
        assert len(train) == 10
        assert eval_ == []

    @pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
    def test_invalid_ratio_raises(self, ratio):
        with pytest.raises(ValueError):
            split_cases(self._make_cases(10), ratio, seed=0)

    def test_empty_train_subset_raises(self):
        with pytest.raises(ValueError):
            split_cases(self._make_cases(3), 0.2, seed=0)


class TestSplitIndicesConsistency:
    @staticmethod
    def _make_cases(n: int) -> list[EvalCase]:
        return [
            EvalCase(
                x=TextGenerationInput(system_prompt=None, user_prompt=f"p{i}"),
                y_true=str(i),
            )
            for i in range(n)
        ]

    def test_indices_match_case_split(self):
        n = 20
        cases = self._make_cases(n)
        train, eval_ = split_cases(cases, 0.7, seed=9)
        train_idx, eval_idx = split_indices(n, 0.7, seed=9)
        assert [c["x"]["user_prompt"] for c in train] == [f"p{i}" for i in train_idx]
        assert [c["x"]["user_prompt"] for c in eval_] == [f"p{i}" for i in eval_idx]
