# Unit tests use obviously synthetic strings shaped like Google's LinkedIn results.
from discovery.parser import clean_snippet, normalise_profile_url, parse_result, parse_title, profile_slug


def test_profile_slug_and_normalisation():
    assert profile_slug("https://in.linkedin.com/in/test-person-123?trk=x") == "test-person-123"
    assert normalise_profile_url("https://in.linkedin.com/in/Test-Person-123/") == "https://www.linkedin.com/in/test-person-123"
    assert profile_slug("https://www.linkedin.com/company/acme") is None
    assert normalise_profile_url("https://example.com/in/someone") is None


def test_parse_title_three_parts():
    t = parse_title("Test Person - Principal - Example Public School | LinkedIn")
    assert t == {"name": "Test Person", "headline": "Principal", "organisation": "Example Public School"}


def test_parse_title_two_parts_role_vs_org():
    assert parse_title("Test Person - Education Consultant | LinkedIn")["headline"] == "Education Consultant"
    assert parse_title("Test Person – Example Group | LinkedIn")["organisation"] == "Example Group"


def test_parse_snippet_fields():
    r = parse_result(
        "Test Person - Director - Example Schools | LinkedIn",
        "Experience: Example Schools · Education: Example University · Location: Gurugram · "
        "500+ connections on LinkedIn. View Test Person's profile on LinkedIn, a professional community of 1 billion members.",
    )
    assert r["location"] == "Gurugram"
    assert r["organisation"] == "Example Schools"
    assert r["connections"] == "500+"
    assert "View" not in r["snippet"]


def test_role_at_org_headline():
    r = parse_result("Test Person - Principal at Example Vidyalaya | LinkedIn", "New Delhi, Delhi, India · Principal")
    assert r["role"] == "Principal"
    assert r["organisation"] == "Example Vidyalaya"
    assert r["location"] == "New Delhi, Delhi, India"


def test_clean_snippet():
    assert clean_snippet("Runs 20 schools. 500+ connections on LinkedIn.") == "Runs 20 schools"
