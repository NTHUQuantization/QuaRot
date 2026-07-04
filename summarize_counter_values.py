#!/usr/bin/env python3
import csv
import glob
import sys


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*_counter_collection.csv"
    for path in sorted(glob.glob(pattern)):
        values = {}
        for row in csv.DictReader(open(path)):
            values[row["Counter_Name"]] = values.get(row["Counter_Name"], 0.0) + float(row["Counter_Value"])
        print(path)
        for name, value in values.items():
            print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
