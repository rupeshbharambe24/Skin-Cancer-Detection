from pathlib import Path

import pandas as pd


def main():
    parts = []
    if Path("binary_results.csv").exists():
        b = pd.read_csv("binary_results.csv")
        b["task"] = "binary"
        parts.append(b)
    if Path("milk10k_results.csv").exists():
        m = pd.read_csv("milk10k_results.csv")
        m["task"] = "multiclass_milk10k"
        parts.append(m)

    if not parts:
        raise SystemExit("No results CSVs found. Run eval scripts first.")

    out = pd.concat(parts, axis=0, ignore_index=True, sort=False)
    out.to_csv("results_tables.csv", index=False)
    print("Wrote: results_tables.csv")


if __name__ == "__main__":
    main()

