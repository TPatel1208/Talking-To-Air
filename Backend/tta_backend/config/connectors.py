"""T30: the connector registry driving the Connectors tab and its endpoints.

One entry per external service a user can link their own credential to.
Adding a connector type is a registry entry, not a schema redesign -- the
list endpoint and the frontend cards are both registry-driven. auth_method
picks the endpoint and form: "token" pastes one token
(PUT /connectors/{type}/token), "email_key" submits an email + API key pair
(PUT /connectors/{type}/credentials).
"""
from __future__ import annotations

CONNECTOR_REGISTRY: list[dict[str, str]] = [
    {
        "connector_type": "earthdata",
        "display_name": "NASA Earthdata Login",
        "auth_method": "token",
        "token_docs_url": "https://urs.earthdata.nasa.gov/documentation/for_users/user_token",
        "docs_link_label": "Generate a token at Earthdata Login",
        "description": (
            "Provides access to NASA Earthdata services"
            ". Paste your Earthdata Login user token below."
        ),
    },
    {
        "connector_type": "epa_aqs",
        "display_name": "EPA AQS",
        "auth_method": "email_key",
        "token_docs_url": "https://aqs.epa.gov/aqsweb/documents/data_api.html#signup",
        "docs_link_label": "Get a free key from EPA AQS",
        "description": (
            "Ground monitor data from the EPA Air Quality System, on your own"
            " request budget. Enter the email you signed up with and the key EPA emailed you."
        ),
    },
]

CONNECTOR_REGISTRY_BY_TYPE: dict[str, dict[str, str]] = {
    entry["connector_type"]: entry for entry in CONNECTOR_REGISTRY
}
