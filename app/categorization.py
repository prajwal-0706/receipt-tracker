from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from app.model_clients import SLMClient

CATEGORIES = [
    "groceries", "dining", "transport", "fuel", "utilities",
    "entertainment", "health", "shopping", "travel", "other",
]

# Keyword -> category. Lowercase, matched as whole word (\b) against item + merchant.
RULES: dict[str, str] = {
    # dining
    "coffee": "dining", "latte": "dining", "espresso": "dining", "cappuccino": "dining",
    "mocha": "dining", "americano": "dining", "tea": "dining", "starbucks": "dining",
    "costa": "dining", "burger": "dining", "pizza": "dining", "sandwich": "dining",
    "restaurant": "dining", "cafe": "dining", "bakery": "dining", "croissant": "dining",
    "mcdonald": "dining", "kfc": "dining", "subway": "dining",
    # groceries
    "milk": "groceries", "bread": "groceries", "eggs": "groceries", "rice": "groceries",
    "vegetables": "groceries", "fruit": "groceries", "cheese": "groceries",
    "walmart": "groceries", "tesco": "groceries", "kroger": "groceries", "aldi": "groceries",
    "safeway": "groceries", "whole foods": "groceries", "trader joe": "groceries",
    # fuel
    "shell": "fuel", "bp": "fuel", "chevron": "fuel", "exxon": "fuel", "petrol": "fuel",
    "diesel": "fuel", "gasoline": "fuel", "unleaded": "fuel",
    # transport
    "uber": "transport", "lyft": "transport", "ola": "transport", "metro": "transport",
    "taxi": "transport", "bus fare": "transport", "parking": "transport",
    # utilities
    "electricity": "utilities", "internet": "utilities", "broadband": "utilities",
    "water bill": "utilities", "gas bill": "utilities",
    # entertainment
    "netflix": "entertainment", "spotify": "entertainment", "cinema": "entertainment",
    "movie": "entertainment", "concert": "entertainment", "ticket": "entertainment",
    # health
    "pharmacy": "health", "cvs": "health", "walgreens": "health", "medicine": "health",
    "doctor": "health", "clinic": "health", "hospital": "health",
    # travel
    "hotel": "travel", "airbnb": "travel", "flight": "travel", "airline": "travel",
    "booking.com": "travel",
    # shopping
    "amazon": "shopping", "target": "shopping", "best buy": "shopping", "ikea": "shopping",
    "shirt": "shopping", "shoes": "shopping",
}


@dataclass
class Categorization:
    category: str
    confidence: float
    source: str  # "rule" | "slm" | "default"


def categorize_item(description: str, merchant: str | None, slm: SLMClient) -> Categorization:
    text = f"{description} {merchant or ''}".lower()
    for keyword, category in RULES.items():
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return Categorization(category, 0.9, "rule")

    prompt = _build_slm_prompt(description, merchant)
    answer = slm.generate(prompt, max_tokens=16).strip().lower()
    for cat in CATEGORIES:
        if cat in answer:
            return Categorization(cat, 0.7, "slm")
    return Categorization("other", 0.3, "default")


def _build_slm_prompt(description: str, merchant: str | None) -> str:
    merchant_clause = f" from merchant '{merchant}'" if merchant else ""
    options = ", ".join(CATEGORIES)
    return (
        f"Classify this receipt line item{merchant_clause} into exactly ONE category.\n"
        f"Item: \"{description}\"\n"
        f"Categories: {options}\n"
        f"Respond with only the category word, nothing else."
    )
