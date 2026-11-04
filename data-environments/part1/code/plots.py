import os
import getpass
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

base_dir = f"/home/{getpass.getuser()}/task1"
results_dir = os.path.join(base_dir, "results")

conn = sqlite3.connect(os.path.join(results_dir, "monitor.db"))
files = pd.read_sql_query("SELECT * FROM files", conn)
calculations = pd.read_sql_query("SELECT * FROM calculations", conn)
conn.close()

files.to_excel(os.path.join(results_dir, "files.xlsx"), index=False)
calculations.to_excel(os.path.join(results_dir, "calculations.xlsx"), index=False)

files["write_duration"] = pd.to_numeric(files["write_duration"], errors="coerce").fillna(0)
files["scale"] = pd.to_numeric(files["scale"], errors="coerce").fillna(0)

calculations["duration"] = pd.to_numeric(calculations["duration"], errors="coerce").fillna(0)
calculations["scale"] = pd.to_numeric(calculations["scale"], errors="coerce").fillna(0)


def make_grouped_bar(df, x_col, group_col, y_col, title, filename):
    plt.figure(figsize=(10, 6))

    if df.empty:
        plt.bar(["no data"], [0])
    else:
        x_values = sorted(df[x_col].unique())
        groups = sorted(df[group_col].unique())
        n_groups = len(groups)
        width = 0.8 / n_groups
        x_positions = range(len(x_values))

        for i, group in enumerate(groups):
            group_df = df[df[group_col] == group].set_index(x_col)
            heights = [group_df[y_col].get(x, 0) for x in x_values]
            offsets = [pos + (i - (n_groups - 1) / 2) * width for pos in x_positions]
            plt.bar(offsets, heights, width=width, label=str(group))

        plt.xticks(list(x_positions), [str(x) for x in x_values])
        plt.legend(title=group_col)

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()


# גרף הכתיבה - קיבוץ עמודות לפי scale, צבע/legend לפי compression+format
# (התיקון לבאג: קודם תוויות זהות של scale+format חפפו זו על זו ומחקו נתונים)
files["format_compression"] = files["compression"].astype(str) + " " + files["format"].astype(str)

# יש 8 שורות (אחת לכל טבלה) לכל צירוף scale+format_compression - מאגדים אותן
# לפני הציור, אחרת make_grouped_bar תקבל כמה שורות לאותו x/group ותשתמש רק באחת מהן
files_agg = (
    files
    .groupby(["scale", "format_compression"])["write_duration"]
    .sum()
    .reset_index()
)

make_grouped_bar(
    files_agg, "scale", "format_compression", "write_duration",
    "Write duration by scale and format", "write.png"
)

# עבור טבלת calculations - חילוץ compression/format משם הקובץ
# (לפי מוסכמת השם ב-generateTPCH.py: {compression}_{table}.{format})
calculations["compression"] = calculations["name"].astype(str).str.split("_").str[0]
calculations["format"] = calculations["name"].astype(str).str.rsplit(".", n=1).str[-1]
calculations["format_compression"] = calculations["compression"] + " " + calculations["format"]

for action in calculations["action"].dropna().unique():
    # פעולות ה-write כבר מוצגות ב-write.png מתוך טבלת files
    if action == "write":
        continue

    data = calculations[calculations["action"] == action]
    make_grouped_bar(
        data, "scale", "format_compression", "duration",
        action, f"{action}.png"
    )

print("DONE")
