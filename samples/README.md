# Samples

Drop the evaluator-provided receipt images directly into this folder.

To run the eval harness, create the two label files below.

## labels.json — for extraction eval

```json
[
  {
    "image": "samples/receipt1.jpg",
    "expected": {
      "merchant": "Blue Bottle Coffee",
      "date": "2025-03-04",
      "total": 11.34,
      "currency": "USD",
      "items": [{ "description": "Latte" }, { "description": "Croissant" }]
    }
  }
]
```

## queries.json — for NL query eval

```json
[
  {
    "q": "how much did I spend on coffee last month?",
    "expected_contains": ["$"]
  },
  {
    "q": "what was my biggest grocery bill in March?",
    "expected_contains": ["$"]
  }
]
```

Both files are gitignored by default — fill them with the evaluator's data before running `python scripts/evaluate.py`.
