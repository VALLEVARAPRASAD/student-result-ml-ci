
import json
import sys

with open("metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]

print("Model accuracy:", round(accuracy, 4))

if accuracy < 0.80:
    print("QUALITY GATE FAILED")
    sys.exit(1)

print("QUALITY GATE PASSED")
