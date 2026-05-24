import argparse
import json
import pandas as pd
from modeling import train_and_select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/salary_profiles.csv")
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    best, results = train_and_select(df, args.output)
    print(json.dumps({"best_model": best, "results": results}, indent=2))


if __name__ == "__main__":
    main()
