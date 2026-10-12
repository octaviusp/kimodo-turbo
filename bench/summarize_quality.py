"""Print the per-prompt rows of a quality run (log lines or out/bench/quality.json) as a table."""

import json
import sys


def rows_from(path: str):
    text = open(path).read().replace("\r", "\n")
    if path.endswith(".json"):
        return json.loads(text)["rows"]
    return [json.loads(line) for line in text.splitlines() if line.startswith('{"prompt"')]


def main() -> None:
    rows = rows_from(sys.argv[1])
    print(f"{'prompt':42s} {'ref↔oracle':>11s} {'turbo↔oracle':>13s} {'turbo↔ref':>10s} {'contacts':>8s} {'turbo s':>7s}")
    for r in rows:
        print(
            f"{r['prompt'][:42]:42s} {r['ref_vs_oracle']['mpjpe_cm']:9.3f}cm {r['turbo_vs_oracle']['mpjpe_cm']:11.3f}cm "
            f"{r['turbo_vs_ref']['mpjpe_cm']:8.3f}cm {r['turbo_vs_ref']['foot_contact_agreement']:8.3f} {r['turbo_seconds']:7.1f}"
        )


if __name__ == "__main__":
    main()
