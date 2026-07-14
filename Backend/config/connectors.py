"""T30: the connector registry driving the Connectors tab and its endpoints.

One entry per external service a user can link their own credential to.
Adding a second connector type later is a registry entry, not a schema
redesign -- the list endpoint and the frontend cards are both registry-driven.
"""
from __future__ import annotations

CONNECTOR_REGISTRY: list[dict[str, str]] = [
    {
        "connector_type": "earthdata",
        "display_name": "NASA Earthdata Login",
        "auth_method": "token",
        "token_docs_url": "https://urs.earthdata.nasa.gov/documentation/for_users/user_token",
        "description": (
            "Paste your Earthdata Login user token so the platform can later act as you "
            "against NASA services, instead of a shared account. Not consumed yet -- "
            "this phase only stores it."
        ),
    },
]

CONNECTOR_REGISTRY_BY_TYPE: dict[str, dict[str, str]] = {
    entry["connector_type"]: entry for entry in CONNECTOR_REGISTRY
}
