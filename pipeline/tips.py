"""Tip text shown on the reveal screen. One tip per breakdown bucket; scam tips live in
scam_templates.yaml. `{pct}` / `{amount}` are filled with this car's own numbers."""
from __future__ import annotations

SOURCES = {
    "FTC": "https://consumer.ftc.gov/articles/buying-used-car-dealer",
    "FTC (online)": "https://consumer.ftc.gov/consumer-alerts/2024/07/what-know-when-buying-used-car-online",
    "FTC (phony sales)": "https://consumer.ftc.gov/consumer-alerts/2019/06/put-brakes-phony-online-car-sales",
    "FTC (gift cards)": "https://consumer.ftc.gov/articles/avoiding-and-reporting-gift-card-scams",
    "NHTSA": "https://www.nhtsa.gov/odometer-fraud",
    "NICB": "https://www.nicb.org/vincheck",
    "NMVTIS": "https://vehiclehistory.bja.ojp.gov/",
}

BUCKET_TIPS: dict[str, tuple[str, str]] = {
    "base": (
        "Year, model and trim set most of a car's price. Compare listings for the exact trim, "
        "and get the out-the-door price in writing before you visit.",
        "FTC",
    ),
    "mileage": (
        "Mileage moved this car's value by {amount} ({pct}). Make sure the odometer matches the "
        "title, service records and wear on the pedals and seats.",
        "NHTSA",
    ),
    "accidents": (
        "The accident history changed this price by {amount} ({pct}). Always get a vehicle "
        "history report and have an independent mechanic inspect the car before you buy.",
        "FTC",
    ),
    "owners": (
        "Owner count shifted the value by {amount} ({pct}). More owners usually means less "
        "complete records, so ask for service history.",
        "FTC",
    ),
    "location": (
        "Where the car is sold changed the price by {amount} ({pct}). Prices vary by state, so "
        "compare listings nearby before deciding a price is fair.",
        "FTC",
    ),
    "title": (
        "A branded title cut {amount} ({pct}) off this car. Check the title history in the "
        "national NMVTIS database before you pay, because some sellers hide salvage brands.",
        "NMVTIS",
    ),
    "private": (
        "Private-party sales are usually cheaper, here {amount} ({pct}), but they're normally "
        "\"as is\" with fewer protections. Get promises in writing and an inspection first.",
        "FTC",
    ),
}


def format_tip(bucket: str, amount: int, reference: int) -> tuple[str, str, str]:
    """Return (tip, source, url) for the bucket with the largest adjustment."""
    text, source = BUCKET_TIPS[bucket]
    sign = "+" if amount >= 0 else "-"
    pct = f"{sign}{abs(amount) / reference:.0%}" if reference else ""
    tip = text.format(amount=f"{sign}${abs(amount):,}", pct=pct)
    return tip, source, SOURCES[source]
