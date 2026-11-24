
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests
import tldextract


def analyze_domain(url: str) -> dict:
    """Retrieve available domain registration information through RDAP."""

    value = url.strip()
    if "://" not in value:
        value = "https://" + value

    try:
        hostname = (urlsplit(value).hostname or "").lower()
    except ValueError:
        hostname = ""

    if not hostname:
        return {
            "domain": None,
            "registration_date": None,
            "domain_age_days": None,
            "registrar": None,
            "status": "Invalid URL or hostname",
        }

    # Extract the registered domain using the Public Suffix List.
    extracted = tldextract.extract(hostname)
    registered_domain = extracted.top_domain_under_public_suffix

    if not registered_domain:
        return {
            "domain": None,
            "registration_date": None,
            "domain_age_days": None,
            "registrar": None,
            "status": "Registered domain could not be determined",
        }

    result = {
        "domain": registered_domain,
        "registration_date": None,
        "domain_age_days": None,
        "registrar": None,
        "status": None,
    }

    endpoint = f"https://rdap.org/domain/{registered_domain}"

    try:
        response = requests.get(
            endpoint,
            timeout=10,
            headers={"Accept": "application/rdap+json"},
        )

        if response.status_code == 404:
            result["status"] = "No RDAP registration record found"
            return result

        response.raise_for_status()
        data = response.json()

        for event in data.get("events", []):
            if event.get("eventAction") == "registration":
                date_text = event.get("eventDate")
                if date_text:
                    registration_date = datetime.fromisoformat(
                        date_text.replace("Z", "+00:00")
                    )

                    if registration_date.tzinfo is None:
                        registration_date = registration_date.replace(
                            tzinfo=timezone.utc
                        )

                    age_days = (
                        datetime.now(timezone.utc) - registration_date
                    ).days

                    result["registration_date"] = (
                        registration_date.isoformat()
                    )
                    result["domain_age_days"] = max(age_days, 0)
                    break

        for entity in data.get("entities", []):
            if "registrar" in entity.get("roles", []):
                for item in entity.get("vcardArray", [None, []])[1]:
                    if (
                        len(item) >= 4
                        and item[0] == "fn"
                        and item[3]
                    ):
                        result["registrar"] = item[3]
                        break

            if result["registrar"]:
                break

        result["status"] = "RDAP lookup successful"

    except requests.Timeout:
        result["status"] = "RDAP request timed out"
    except requests.RequestException as exc:
        result["status"] = f"RDAP request failed: {exc}"
    except (ValueError, TypeError, KeyError, IndexError):
        result["status"] = "Could not parse RDAP response"

    return result


if __name__ == "__main__":
    url = input("Enter a URL for domain registration analysis: ").strip()
    result = analyze_domain(url)

    print("\nDomain Registration Analysis")
    print("-" * 35)

    for key, value in result.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print(
        "\nNote: Missing registration data does not automatically "
        "mean a domain is phishing."
    )
