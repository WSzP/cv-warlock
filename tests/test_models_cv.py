"""Tests for CV data models and validators."""

from cv_warlock.models.cv import (
    Certification,
    ContactInfo,
    CVData,
    Education,
    Experience,
    Project,
    _coerce_to_list,
    _coerce_to_model_list,
)


class TestCoerceToList:
    """Tests for the _coerce_to_list helper function."""

    def test_none_returns_empty_list(self) -> None:
        assert _coerce_to_list(None) == []

    def test_empty_string_returns_empty_list(self) -> None:
        assert _coerce_to_list("") == []

    def test_list_passthrough(self) -> None:
        result = _coerce_to_list(["Python", "JavaScript"])
        assert result == ["Python", "JavaScript"]

    def test_list_converts_items_to_strings(self) -> None:
        result = _coerce_to_list([1, 2, 3])
        assert result == ["1", "2", "3"]

    def test_json_array_string(self) -> None:
        result = _coerce_to_list('["Python", "JavaScript", "TypeScript"]')
        assert result == ["Python", "JavaScript", "TypeScript"]

    def test_json_array_with_whitespace(self) -> None:
        result = _coerce_to_list('  ["Python", "JavaScript"]  ')
        assert result == ["Python", "JavaScript"]

    def test_comma_separated_string(self) -> None:
        result = _coerce_to_list("Python, JavaScript, TypeScript")
        assert result == ["Python", "JavaScript", "TypeScript"]

    def test_comma_separated_strips_whitespace(self) -> None:
        result = _coerce_to_list("  Python  ,  JavaScript  ,  TypeScript  ")
        assert result == ["Python", "JavaScript", "TypeScript"]

    def test_single_value_string(self) -> None:
        result = _coerce_to_list("Python")
        assert result == ["Python"]

    def test_invalid_json_falls_back_to_comma_split(self) -> None:
        # Malformed JSON that starts with [ but isn't valid
        result = _coerce_to_list("[Python, JavaScript")
        assert result == ["[Python", "JavaScript"]

    def test_non_string_non_list_converts_to_string(self) -> None:
        result = _coerce_to_list(42)
        assert result == ["42"]


class TestCoerceToModelList:
    """Tests for the _coerce_to_model_list helper function."""

    def test_none_returns_empty_list(self) -> None:
        assert _coerce_to_model_list(None) == []

    def test_list_passthrough(self) -> None:
        data = [{"name": "test"}]
        assert _coerce_to_model_list(data) == data

    def test_json_array_string(self) -> None:
        result = _coerce_to_model_list('[{"name": "Project A"}, {"name": "Project B"}]')
        assert result == [{"name": "Project A"}, {"name": "Project B"}]

    def test_json_array_with_whitespace(self) -> None:
        result = _coerce_to_model_list('  [{"name": "test"}]  ')
        assert result == [{"name": "test"}]

    def test_invalid_json_returns_empty_list(self) -> None:
        result = _coerce_to_model_list("[invalid json")
        assert result == []

    def test_non_array_string_returns_empty_list(self) -> None:
        result = _coerce_to_model_list("just a string")
        assert result == []

    def test_non_string_non_list_returns_empty_list(self) -> None:
        result = _coerce_to_model_list(42)
        assert result == []


class TestContactInfo:
    """Tests for ContactInfo model."""

    def test_minimal_contact(self) -> None:
        contact = ContactInfo(name="John Doe")
        assert contact.name == "John Doe"
        assert contact.email is None
        assert contact.phone is None

    def test_full_contact(self) -> None:
        contact = ContactInfo(
            name="John Doe",
            email="john@example.com",
            phone="+1234567890",
            location="New York",
            linkedin="linkedin.com/in/johndoe",
            github="github.com/johndoe",
            website="johndoe.com",
        )
        assert contact.name == "John Doe"
        assert contact.email == "john@example.com"
        assert contact.linkedin == "linkedin.com/in/johndoe"


class TestExperience:
    """Tests for Experience model."""

    def test_minimal_experience(self) -> None:
        exp = Experience(
            title="Software Engineer",
            company="Tech Corp",
            start_date="January 2020",
        )
        assert exp.title == "Software Engineer"
        assert exp.end_date is None
        assert exp.achievements == []
        assert exp.skills_used == []

    def test_experience_with_lists(self) -> None:
        exp = Experience(
            title="Software Engineer",
            company="Tech Corp",
            start_date="January 2020",
            end_date="December 2022",
            achievements=["Led team of 5", "Increased efficiency by 20%"],
            skills_used=["Python", "AWS"],
        )
        assert len(exp.achievements) == 2
        assert "Python" in exp.skills_used

    def test_experience_coerces_achievements_from_string(self) -> None:
        exp = Experience(
            title="Software Engineer",
            company="Tech Corp",
            start_date="January 2020",
            achievements="Led team, Improved performance",
        )
        assert exp.achievements == ["Led team", "Improved performance"]

    def test_experience_coerces_achievements_from_json(self) -> None:
        exp = Experience(
            title="Software Engineer",
            company="Tech Corp",
            start_date="January 2020",
            achievements='["Achievement 1", "Achievement 2"]',
        )
        assert exp.achievements == ["Achievement 1", "Achievement 2"]


class TestEducation:
    """Tests for Education model."""

    def test_minimal_education(self) -> None:
        edu = Education(
            degree="B.S. Computer Science",
            institution="MIT",
            graduation_date="May 2019",
        )
        assert edu.degree == "B.S. Computer Science"
        assert edu.gpa is None
        assert edu.relevant_coursework == []

    def test_education_coerces_coursework(self) -> None:
        edu = Education(
            degree="B.S. Computer Science",
            institution="MIT",
            graduation_date="May 2019",
            relevant_coursework="Algorithms, Data Structures, Machine Learning",
        )
        assert edu.relevant_coursework == ["Algorithms", "Data Structures", "Machine Learning"]


class TestProject:
    """Tests for Project model."""

    def test_minimal_project(self) -> None:
        proj = Project(name="My Project", description="A cool project")
        assert proj.name == "My Project"
        assert proj.technologies == []
        assert proj.url is None

    def test_project_coerces_technologies(self) -> None:
        proj = Project(
            name="My Project",
            description="A cool project",
            technologies='["Python", "FastAPI", "PostgreSQL"]',
        )
        assert proj.technologies == ["Python", "FastAPI", "PostgreSQL"]


class TestCertification:
    """Tests for Certification model."""

    def test_minimal_certification(self) -> None:
        cert = Certification(name="AWS Solutions Architect", issuer="Amazon")
        assert cert.name == "AWS Solutions Architect"
        assert cert.date is None


class TestCVData:
    """Tests for CVData model."""

    def test_minimal_cv_data(self) -> None:
        cv = CVData(contact=ContactInfo(name="John Doe"))
        assert cv.contact.name == "John Doe"
        assert cv.summary is None
        assert cv.experiences == []
        assert cv.skills == []

    def test_cv_data_coerces_skills(self) -> None:
        cv = CVData(
            contact=ContactInfo(name="John Doe"),
            skills="Python, JavaScript, TypeScript",
        )
        assert cv.skills == ["Python", "JavaScript", "TypeScript"]

    def test_cv_data_coerces_languages(self) -> None:
        cv = CVData(
            contact=ContactInfo(name="John Doe"),
            languages='["English", "Spanish", "French"]',
        )
        assert cv.languages == ["English", "Spanish", "French"]

    def test_cv_data_coerces_experiences_from_json(self) -> None:
        cv = CVData(
            contact=ContactInfo(name="John Doe"),
            experiences='[{"title": "Engineer", "company": "Corp", "start_date": "2020"}]',
        )
        assert len(cv.experiences) == 1
        assert cv.experiences[0].title == "Engineer"

    def test_cv_data_with_nested_models(self) -> None:
        cv = CVData(
            contact=ContactInfo(name="John Doe", email="john@example.com"),
            summary="Experienced engineer",
            experiences=[
                Experience(
                    title="Senior Engineer",
                    company="Tech Corp",
                    start_date="2020",
                    achievements=["Led projects"],
                )
            ],
            education=[
                Education(
                    degree="M.S. Computer Science",
                    institution="Stanford",
                    graduation_date="2019",
                )
            ],
            skills=["Python", "AWS", "Kubernetes"],
        )
        assert cv.contact.email == "john@example.com"
        assert len(cv.experiences) == 1
        assert len(cv.education) == 1
        assert len(cv.skills) == 3

    def test_to_scoring_dict_strips_pii(self) -> None:
        """Test that to_scoring_dict removes PII but keeps name."""
        cv = CVData(
            contact=ContactInfo(
                name="John Doe",
                email="john@secret.com",
                phone="+1234567890",
                linkedin="linkedin.com/in/johndoe",
                github="github.com/johndoe",
            ),
            summary="Experienced engineer",
            skills=["Python", "AWS"],
        )
        scoring_dict = cv.to_scoring_dict()

        # Name should be preserved for personalization
        assert scoring_dict["contact"]["name"] == "John Doe"

        # PII should be stripped
        assert "email" not in scoring_dict["contact"]
        assert "phone" not in scoring_dict["contact"]
        assert "linkedin" not in scoring_dict["contact"]
        assert "github" not in scoring_dict["contact"]

        # Other fields should be preserved
        assert scoring_dict["summary"] == "Experienced engineer"
        assert scoring_dict["skills"] == ["Python", "AWS"]

    def test_to_scoring_json_is_valid_json(self) -> None:
        """Test that to_scoring_json returns valid JSON without PII."""
        import json

        cv = CVData(
            contact=ContactInfo(
                name="Jane Smith",
                email="jane@company.com",
                phone="+9876543210",
            ),
            skills=["TypeScript", "React"],
        )
        scoring_json = cv.to_scoring_json()

        # Should be valid JSON
        parsed = json.loads(scoring_json)

        # Name preserved, PII stripped
        assert parsed["contact"]["name"] == "Jane Smith"
        assert "email" not in parsed["contact"]
        assert "phone" not in parsed["contact"]

    def test_to_scoring_json_respects_indent(self) -> None:
        """Test that to_scoring_json respects indent parameter."""
        cv = CVData(contact=ContactInfo(name="Test User"))

        # Different indent values should produce different output
        json_indent_2 = cv.to_scoring_json(indent=2)
        json_indent_4 = cv.to_scoring_json(indent=4)

        # More indent = longer output
        assert len(json_indent_4) > len(json_indent_2)
