"""Tests for chain-of-thought reasoning models' input coercion."""

from cv_warlock.models.reasoning import SkillsReasoning


class TestSkillsReasoningCategoryGroupings:
    """Tests for SkillsReasoning.category_groupings validator."""

    def test_accepts_actual_lists(self) -> None:
        reasoning = SkillsReasoning(
            category_groupings={"Languages": ["Python", "Go"], "Cloud": ["AWS"]}
        )
        assert reasoning.category_groupings == {"Languages": ["Python", "Go"], "Cloud": ["AWS"]}

    def test_coerces_comma_separated_string(self) -> None:
        """Some LLMs return 'Python, Go' instead of ['Python', 'Go'] per category."""
        reasoning = SkillsReasoning(
            category_groupings={
                "Core ML/NLP & LLM": "PyTorch, Scikit-learn, spaCy, Explainable AI (XAI)",
                "Cloud & Data Platforms": "Amazon Web Services, Azure",
            }
        )
        assert reasoning.category_groupings == {
            "Core ML/NLP & LLM": ["PyTorch", "Scikit-learn", "spaCy", "Explainable AI (XAI)"],
            "Cloud & Data Platforms": ["Amazon Web Services", "Azure"],
        }

    def test_coerces_json_stringified_list(self) -> None:
        reasoning = SkillsReasoning(category_groupings={"Languages": '["Python", "Go"]'})
        assert reasoning.category_groupings == {"Languages": ["Python", "Go"]}

    def test_empty_string_yields_empty_list(self) -> None:
        reasoning = SkillsReasoning(category_groupings={"Languages": ""})
        assert reasoning.category_groupings == {"Languages": []}

    def test_default_is_empty_dict(self) -> None:
        reasoning = SkillsReasoning()
        assert reasoning.category_groupings == {}
