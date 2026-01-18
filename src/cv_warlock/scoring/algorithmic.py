"""Algorithmic scoring for CV-job matching.

Computes deterministic, reproducible sub-scores for various matching criteria.
Uses exact string matching only - no embeddings required.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cv_warlock.models.cv import CVData
    from cv_warlock.models.job_spec import JobRequirements

from cv_warlock.scoring.models import AlgorithmicScores

logger = logging.getLogger(__name__)


class AlgorithmicScorer:
    """Compute deterministic, reproducible algorithmic scores.

    Calculates multiple sub-scores for CV-job matching including exact skill
    matches, experience years, education, and recency. Uses exact string
    matching only - no embeddings or semantic similarity.
    """

    # Knockout rule uses exact string matching only (case-insensitive)
    # No semantic/embedding matching - if the skill isn't explicitly listed, it's missing
    # This is the safest approach for ATS accuracy

    # Score weights for combining sub-scores
    # FUTURE: These weights can be made configurable per job type
    WEIGHTS = {
        "exact_skill_match": 0.35,  # Weight for exact string matches (increased)
        "experience_years_fit": 0.25,  # Weight for experience years
        "education_match": 0.15,  # Weight for education level
        "recency_score": 0.25,  # Weight for recent experience relevance
    }

    def compute(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
    ) -> AlgorithmicScores:
        """Compute all algorithmic sub-scores.

        Args:
            cv_data: Parsed CV data.
            job_requirements: Parsed job requirements.

        Returns:
            AlgorithmicScores with all sub-scores and knockout status.
        """
        # Compute individual sub-scores
        exact_match = self._compute_exact_match(cv_data, job_requirements)
        years_fit = self._compute_years_fit(cv_data, job_requirements)
        edu_match = self._compute_education_match(cv_data, job_requirements)
        recency = self._compute_recency(cv_data, job_requirements)

        # Check knockout rule
        knockout, reason = self.check_knockout(cv_data, job_requirements)

        # Compute weighted total (0 if knockout triggered)
        if knockout:
            total = 0.0
        else:
            total = (
                self.WEIGHTS["exact_skill_match"] * exact_match
                + self.WEIGHTS["experience_years_fit"] * years_fit
                + self.WEIGHTS["education_match"] * edu_match
                + self.WEIGHTS["recency_score"] * recency
            )

        return AlgorithmicScores(
            exact_skill_match=exact_match,
            semantic_skill_match=exact_match,  # Same as exact (no semantic matching)
            document_similarity=exact_match,  # Same as exact (no embeddings)
            experience_years_fit=years_fit,
            education_match=edu_match,
            recency_score=recency,
            total=total,
            knockout_triggered=knockout,
            knockout_reason=reason,
        )

    def check_knockout(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
    ) -> tuple[bool, str | None]:
        """Check if required skills are missing (knockout rule).

        A candidate is "knocked out" (auto-fail) if ANY required skill
        is not found via exact string matching (case-insensitive).

        No semantic/embedding matching is used - this ensures ATS accuracy.
        If a skill isn't explicitly listed in the CV, it counts as missing.

        Args:
            cv_data: Parsed CV data.
            job_requirements: Parsed job requirements.

        Returns:
            Tuple of (knockout_triggered, reason_message).
        """
        required_skills = job_requirements.required_skills
        if not required_skills:
            return False, None

        # Collect all CV skills (explicit + from experiences)
        cv_skills = self._collect_cv_skills(cv_data)
        cv_skills_lower = {s.lower() for s in cv_skills}

        # Also check experience descriptions for keyword mentions
        exp_text = self._get_experience_text(cv_data).lower()

        missing_required: list[str] = []
        for req_skill in required_skills:
            skill_lower = req_skill.lower()

            # Check exact match in skills list (case-insensitive)
            if skill_lower in cv_skills_lower:
                continue

            # Check if mentioned in experience text (case-insensitive)
            if skill_lower in exp_text:
                continue

            # No semantic matching - skill is missing
            missing_required.append(req_skill)

        if missing_required:
            return True, f"Missing required skills: {', '.join(missing_required)}"
        return False, None

    def _collect_cv_skills(self, cv_data: CVData) -> set[str]:
        """Collect all skills from CV (explicit + from experiences)."""
        skills = set(cv_data.skills)
        for exp in cv_data.experiences:
            skills.update(exp.skills_used)
        return skills

    def _get_experience_text(self, cv_data: CVData) -> str:
        """Get concatenated experience text for keyword search."""
        parts: list[str] = []
        for exp in cv_data.experiences:
            parts.append(exp.title)
            parts.append(exp.description)
            parts.extend(exp.achievements)
        return " ".join(parts)

    def _compute_exact_match(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
    ) -> float:
        """Compute exact skill match percentage."""
        cv_skills = {s.lower() for s in self._collect_cv_skills(cv_data)}

        # Also include skills mentioned in experience text
        exp_text = self._get_experience_text(cv_data).lower()

        required = job_requirements.required_skills
        preferred = job_requirements.preferred_skills

        # Count matches (exact or mentioned in text)
        required_matches = 0
        for skill in required:
            skill_lower = skill.lower()
            if skill_lower in cv_skills or skill_lower in exp_text:
                required_matches += 1

        preferred_matches = 0
        for skill in preferred:
            skill_lower = skill.lower()
            if skill_lower in cv_skills or skill_lower in exp_text:
                preferred_matches += 1

        # Calculate scores
        required_score = required_matches / len(required) if required else 1.0
        preferred_score = preferred_matches / len(preferred) if preferred else 1.0

        # Weighted combination (required worth more)
        return 0.7 * required_score + 0.3 * preferred_score

    def _compute_years_fit(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
    ) -> float:
        """Compute experience years fit score."""
        required_years = job_requirements.required_experience_years
        if required_years is None or required_years == 0:
            return 1.0  # No requirement

        # Calculate candidate's total years
        total_years = 0.0
        current_year = datetime.now().year

        for exp in cv_data.experiences:
            start_year = self._extract_year(exp.start_date)
            end_year = self._extract_year(exp.end_date) or current_year

            if start_year and start_year <= end_year:
                total_years += end_year - start_year

        # Score based on how well years match requirement
        if total_years >= required_years:
            return 1.0
        elif total_years >= required_years * 0.7:
            return 0.8  # Close enough (within 30%)
        elif total_years >= required_years * 0.5:
            return 0.5  # Partial match
        else:
            return max(0.2, total_years / required_years)

    def _compute_education_match(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
    ) -> float:
        """Compute education level match score."""
        required_edu = job_requirements.required_education
        if not required_edu:
            return 1.0  # No requirement

        required_lower = required_edu.lower()

        # Education level hierarchy
        edu_levels = {
            "phd": 5,
            "doctorate": 5,
            "doctoral": 5,
            "master": 4,
            "mba": 4,
            "msc": 4,
            "ms": 4,
            "ma": 4,
            "bachelor": 3,
            "bsc": 3,
            "bs": 3,
            "ba": 3,
            "undergraduate": 3,
            "associate": 2,
            "diploma": 2,
            "high school": 1,
            "secondary": 1,
            "ged": 1,
        }

        # Find required level
        required_level = 0
        for edu_key, level in edu_levels.items():
            if edu_key in required_lower:
                required_level = max(required_level, level)

        if required_level == 0:
            return 1.0  # Could not determine requirement

        # Find candidate's highest level
        candidate_level = 0
        for edu in cv_data.education:
            degree_lower = edu.degree.lower()
            for edu_key, level in edu_levels.items():
                if edu_key in degree_lower:
                    candidate_level = max(candidate_level, level)

        # Score based on level match
        if candidate_level >= required_level:
            return 1.0
        elif candidate_level == required_level - 1:
            return 0.7  # One level below
        elif candidate_level > 0:
            return 0.4  # Has some education
        else:
            return 0.2  # No matching education found

    def _compute_recency(
        self,
        cv_data: CVData,
        job_requirements: JobRequirements,
    ) -> float:
        """Compute recency-weighted experience relevance.

        Enterprise-standard recency scoring that considers:
        1. Per-experience skills and text mentions
        2. Overall CV skills applied to recent experiences (within 2 years)
           - Rationale: If someone lists Python as a skill and is currently employed,
             they're likely using it in their current role even if not explicitly stated
        3. Exponential decay weighting for older experiences
        """
        if not cv_data.experiences:
            return 0.0

        current_year = datetime.now().year

        # Target skills for relevance check
        target_skills = {
            s.lower() for s in job_requirements.required_skills + job_requirements.preferred_skills
        }

        if not target_skills:
            return 1.0  # No target skills to check

        # Overall CV skills (used for recent experience boost)
        overall_cv_skills = {s.lower() for s in cv_data.skills}

        weighted_scores: list[float] = []

        for exp in cv_data.experiences:
            # Calculate recency weight (exponential decay)
            end_year = self._extract_year(exp.end_date) or current_year
            years_ago = max(0, current_year - end_year)
            # Decay by ~15% per year (half-life ~4.6 years)
            recency_weight = max(0.1, pow(0.85, years_ago))

            # Calculate relevance of this experience
            exp_skills = {s.lower() for s in exp.skills_used}

            # For RECENT experiences (within 2 years), also consider overall CV skills
            # Rationale: If someone lists Python as a top skill and is currently employed,
            # they're almost certainly using it in their current role
            if years_ago <= 2:
                exp_skills = exp_skills | overall_cv_skills

            exp_text = f"{exp.title} {exp.description} {' '.join(exp.achievements)}".lower()

            # Check skill overlap
            skill_overlap = len(exp_skills & target_skills)

            # Check keyword mentions in text
            keyword_mentions = sum(1 for skill in target_skills if skill in exp_text)

            # Combine into relevance score
            relevance = min(
                1.0, (skill_overlap + keyword_mentions * 0.5) / max(len(target_skills), 1)
            )

            weighted_scores.append(relevance * recency_weight)

        # Average weighted scores, scaled up slightly
        if not weighted_scores:
            return 0.0
        return min(1.0, sum(weighted_scores) / len(weighted_scores) * 1.5)

    @staticmethod
    def _extract_year(date_str: str | None) -> int | None:
        """Extract year from date string."""
        if not date_str:
            return None

        # Handle "Present", "Current", etc.
        if any(word in date_str.lower() for word in ["present", "current", "now", "ongoing"]):
            return datetime.now().year

        # Try to find 4-digit year
        match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if match:
            return int(match.group())

        return None
