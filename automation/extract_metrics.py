#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the RouterArena project
# SPDX-License-Identifier: Apache-2.0

"""Extract metrics from evaluation output and save to JSON file."""

import json
import re
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: extract_metrics.py <output_file>")
        sys.exit(1)

    output_file = sys.argv[1]

    try:
        with open(output_file, "r") as f:
            content = f.read()

        # Find the line that starts the Metrics JSON block
        match = re.search(r"Metrics:\s*\{", content)
        if match:
            start_idx = match.start()
            # Find the first '{' after "Metrics:"
            brace_start = content.find("{", match.end() - 1)
            if brace_start == -1:
                raise ValueError("Could not find '{' after 'Metrics:'")

            # Manually parse a balanced-brace JSON object so nested objects work
            brace_count = 0
            end_idx = None
            for idx in range(brace_start, len(content)):
                ch = content[idx]
                if ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = idx + 1
                        break

            if end_idx is None:
                raise ValueError("Unbalanced braces while parsing Metrics JSON block")

            metrics_json = content[brace_start:end_idx]
            metrics = json.loads(metrics_json)

            # Write metrics to file
            with open("metrics.json", "w") as mf:
                json.dump(metrics, mf)

            # Output key metrics as step outputs
            print(f"accuracy={metrics['accuracy']}")
            print(f"arena_score={metrics['arena_score']}")
            print(f"total_cost={metrics['total_cost']}")
            print(f"num_queries={metrics['num_queries']}")
        else:
            print("No metrics found in output", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error extracting metrics: {e}", file=sys.stderr)
        sys.exit(1)
