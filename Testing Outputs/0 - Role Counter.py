import json
import pandas as pd

#=======parameters to change=========
ministers_json = r"G:\My Drive\Birkbeck\Project\Hansard\Source data\ministers-2010.json"
output_csv = r"G:\My Drive\Birkbeck\Project\Hansard\distinct_roles.csv"

window_start = "2015-01-01"                #only posts live during the corpus window can affect tiering
window_end = "2020-12-31"
skip_sources = ["datadotparl/committee"]   #same exclusion as script 2

#==========load============
ministers = json.load(open(ministers_json, encoding="utf-8"))
memberships = ministers["memberships"]

print(len(memberships), "membership records loaded")

#==========date padding============
#mySociety sometimes stores truncated dates ("2015", "2015-05"). Script 2 compares dates
#as plain strings, so a short date would sort wrongly against a full one. Pad to 10 chars
#in the direction that keeps the spell as wide as possible, so nothing is dropped by accident.
def pad_date(value, is_end):
    if value is None or value == "":
        return "3000-12-31" if is_end else "1000-01-01"
    if len(value) == 10:
        return value
    if len(value) == 7:                       #"2015-05"
        return value + "-31" if is_end else value + "-01"
    if len(value) == 4:                       #"2015"
        return value + "-12-31" if is_end else value + "-01-01"
    return value

short_dates = 0

#==========collect roles============
#keyed on (source, role) so the same title appearing under two sources stays separate
role_rows = {}

for record in memberships:
    source = record["source"]
    if source in skip_sources:
        continue

    raw_start = record["start_date"]
    raw_end = record.get("end_date")

    if len(str(raw_start)) != 10 or (raw_end is not None and raw_end != "" and len(str(raw_end)) != 10):
        short_dates += 1

    start = pad_date(raw_start, is_end=False)
    end = pad_date(raw_end, is_end=True)

    if end < window_start or start > window_end:   #spell ended before, or began after, the corpus
        continue

    key = (source, record["role"])

    if key not in role_rows:
        role_rows[key] = {"source": source,
                          "role": record["role"],
                          "spells": 0,
                          "people": set(),
                          "earliest": start,
                          "latest": end}

    role_rows[key]["spells"] += 1
    role_rows[key]["people"].add(record["person_id"])

    if start < role_rows[key]["earliest"]:
        role_rows[key]["earliest"] = start
    if end > role_rows[key]["latest"]:
        role_rows[key]["latest"] = end

#==========checks============
if len(role_rows) == 0:
    raise SystemExit("no roles found - check the path and the source names")

print(short_dates, "records with a truncated date (padded, not dropped)")
print(len(role_rows), "distinct role titles live in", window_start, "to", window_end)

#==========build table============
table_rows = []
for key in role_rows:
    entry = role_rows[key]
    table_rows.append({"source": entry["source"],
                       "role": entry["role"],
                       "spells": entry["spells"],
                       "people": len(entry["people"]),
                       "earliest": entry["earliest"],
                       "latest": entry["latest"]})

roles = pd.DataFrame(table_rows)
roles = roles.sort_values(["source", "spells", "role"], ascending=[True, False, True]).reset_index(drop=True)

#==========print by source============
for source in sorted(roles["source"].unique()):
    subset = roles[roles["source"] == source]
    print("")
    print("=" * 60)
    print(source, "-", len(subset), "distinct titles")
    print("=" * 60)
    for position in range(len(subset)):
        row = subset.iloc[position]
        print(str(row["spells"]).rjust(4), "spells |", str(row["people"]).rjust(3), "people |", row["role"])

#==========save============
roles.to_csv(output_csv, index=False, encoding="utf-8")
print("")
print("saved:", output_csv)