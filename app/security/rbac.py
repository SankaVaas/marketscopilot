"""
Entitlement rules: which document classifications each role may retrieve.

This is intentionally simple (a static dict) for the demo. In production this
would be backed by an entitlements service / data-catalog integration, but the
interface (`allowed_classifications(role)`) stays the same.
"""

ROLE_ENTITLEMENTS: dict[str, set[str]] = {
    "front_office_analyst": {"public", "front_office_only"},
    "compliance_officer": {"public", "front_office_only", "compliance_restricted"},
    "public_reader": {"public"},
}


def allowed_classifications(role: str) -> set[str]:
    return ROLE_ENTITLEMENTS.get(role, {"public"})


def is_authorized(role: str, classification: str) -> bool:
    return classification in allowed_classifications(role)
