import csv
from pathlib import Path

data_dir = Path("/workspaces/fof8-scout/data-generation/data/raw/DRAFT003")

# Check a smaller year
year = 2050
with open(data_dir / str(year) / "player_record.csv") as f:
    reader = csv.DictReader(f)
    rows = []
    for row in reader:
        sal = int(row["Salary_Year_1"])
        bon = int(row["Bonus_Year_1"])
        rows.append(
            {
                "pid": row["Player_ID"],
                "pos": row["Position_Group"],
                "exp": row["Experience"],
                "sal": sal,
                "bon": bon,
                "total": sal + bon,
            }
        )

# Sort by salary desc
rows.sort(key=lambda x: x["sal"], reverse=True)
print(f"=== Top 15 salaries in {year} ===")
print("Cap = 33720 (tens of thousands) = $337,200,000")
for r in rows[:15]:
    cap_pct_if_thousands = r["sal"] * 1000 / 337_200_000 * 100
    cap_pct_if_tens = r["sal"] * 10_000 / 337_200_000 * 100
    print(
        f"  PID={r['pid']:>7}, Pos={r['pos']:>3}, Exp={r['exp']:>2}, "
        f"Sal={r['sal']:>5}, Bon={r['bon']:>5} | "
        f"if thousands: ${r['sal'] * 1000:>12,} ({cap_pct_if_thousands:.1f}%) | "
        f"if tens: ${r['sal'] * 10_000:>12,} ({cap_pct_if_tens:.1f}%)"
    )

# Minimum salary players
min_sal_rows = [r for r in rows if r["sal"] > 0]
min_sal_rows.sort(key=lambda x: x["sal"])
print(f"\n=== Bottom 5 salaries (nonzero) in {year} ===")
for r in min_sal_rows[:5]:
    print(
        f"  PID={r['pid']:>7}, Pos={r['pos']:>3}, Exp={r['exp']:>2}, "
        f"Sal={r['sal']:>5}, Bon={r['bon']:>5}"
    )

print(f"\nMin nonzero salary = {min_sal_rows[0]['sal']}")
print(f"  If thousands: ${min_sal_rows[0]['sal'] * 1000:,}")
print("  Universe minimum salary (in thousands) = 104 → $104,000")
print(f"  Match? {min_sal_rows[0]['sal'] == 104}")
