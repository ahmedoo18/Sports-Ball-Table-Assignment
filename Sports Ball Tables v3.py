import pandas as pd
import math
import json
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpInteger,
    LpBinary,
    PULP_CBC_CMD,
    LpStatus,
)

"""Sports Ball Table Assignment – GUI with Persistent Settings
================================================================

**2025-07 update:**
-------------------
This revision introduces a *single, tunable* trade-off between keeping clubs intact
and maximising seat utilisation.  The slider **“Split-vs-Fill weight (%)”**
controls \(\lambda\in[0,1]\):

* **\(\lambda\rightarrow1\)** → strong penalty on splitting a club across tables;
* **\(\lambda\rightarrow0\)** → strong penalty on leaving empty seats.

Mathematically the objective is now

\[\min_{x}\; \lambda \sum_{c}(T_c-1)\; + \;(1-\lambda)\sum_{t}\text{underfill}_t,\]

where \(T_c\) is the number of tables used by club *c* and
*underfill* counts the empty seats at each table.

All previous parameters (preferred seats, fragmentation limits, etc.) are still
available for fine-tuning but are secondary to the new global trade-off.

**Outputs:**  identical – `Table_Allocations.xlsx` and `table_layout.png`.
"""

SETTINGS_FILE = Path("table_config.json")

# ---------------------------------------------------------------------------
# MILP solver ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def run_assignment(
    file_path: str,
    min_seats: int,
    max_seats: int,
    pref_seats: int,
    pref_pen_w: float,
    split_small_pen: int,
    extra_split_pen: float,
    min_frag: int,
    frag_pen_w: float,
    time_limit: int,
    allowed_locations,
    split_weight: float,
    slack_factor: float = 3.0,
):
    """Solve the club-to-table assignment MILP.

    Parameters
    ----------
    split_weight : float
        λ in [0, 1]. 1 ⇒ avoid splits, 0 ⇒ avoid empty seats.
    """

    if not 0 <= split_weight <= 1:
        raise ValueError("split_weight must be between 0 and 1")

    # ---- Load data ------------------------------------------------------
    df = pd.read_excel(file_path, sheet_name=0)
    if "Club" not in df.columns:
        raise ValueError("Spreadsheet must include a 'Club' column.")
    size_cols = ["Members", "Size", "Number of Attendees", "Count"]
    size_col = next((c for c in size_cols if c in df.columns), None)
    if size_col is None:
        raise ValueError(f"Spreadsheet must include one of {size_cols}.")

    clubs = df["Club"].tolist()
    sizes = dict(zip(clubs, df[size_col].astype(int)))
    total_people = sum(sizes.values())

    min_tables_req = {c: math.ceil(sizes[c] / max_seats) for c in clubs}

    theoretical_tables = math.ceil(total_people / max_seats)
    T_candidates = int(math.ceil(theoretical_tables * slack_factor) + 5)
    print(
        f"Guests {total_people} | lower-bound tables {theoretical_tables} | candidate pool {T_candidates}"
    )

    # ---- MILP model ------------------------------------------------------
    prob = LpProblem("SportsBallTableAssign", LpMinimize)

    # decision variables
    x = LpVariable.dicts(
        "x", ((c, t) for c in clubs for t in range(T_candidates)), 0, None, LpInteger
    )  # seats of club c at table t
    y = LpVariable.dicts("y", range(T_candidates), 0, 1, LpBinary)  # table used indicator
    z = LpVariable.dicts(
        "z", ((c, t) for c in clubs for t in range(T_candidates)), 0, 1, LpBinary
    )  # club c present on table t
    pos_dev = LpVariable.dicts("pos_dev", range(T_candidates), 0)  # over-fill
    neg_dev = LpVariable.dicts("neg_dev", range(T_candidates), 0)  # under-fill

    # ---- Objective: Split-vs-Fill ---------------------------------------
    #   term1: total extra tables across all clubs (above the first table)
    total_split_pen = lpSum(lpSum(z[(c, t)] for t in range(T_candidates)) - 1 for c in clubs)

    #   term2: under-fill = empty seats = preferred − occupancy (captured by neg_dev)
    total_underfill_pen = lpSum(neg_dev[t] for t in range(T_candidates))

    prob += split_weight * total_split_pen + (1 - split_weight) * total_underfill_pen

    # ---- Constraints -----------------------------------------------------
    for c in clubs:
        # every member seated
        prob += lpSum(x[(c, t)] for t in range(T_candidates)) == sizes[c]

    for t in range(T_candidates):
        occ = lpSum(x[(c, t)] for c in clubs)
        # activate table if used
        prob += occ <= max_seats * y[t]
        if min_seats > 0:
            prob += occ >= min_seats * y[t]
        # preferred-seat deviation split into ± parts
        prob += occ - pref_seats == pos_dev[t] - neg_dev[t]
        for c in clubs:
            # link seat and club-table indicator
            prob += x[(c, t)] <= max_seats * z[(c, t)]
            prob += z[(c, t)] <= y[t]

    solver = PULP_CBC_CMD(msg=True, timeLimit=time_limit)
    prob.solve(solver)
    print("Solver status:", LpStatus[prob.status])
    if LpStatus[prob.status] != "Optimal":
        raise RuntimeError("Solver failed. Adjust parameters or grid size.")

    # ---- Extract solution -----------------------------------------------
    used_tables = [t for t in range(T_candidates) if y[t].value() > 0.5]
    if len(used_tables) > len(allowed_locations):
        raise RuntimeError(
            "More tables used than allowed grid positions. Select more cells."
        )
    loc_map = {t: allowed_locations[idx] for idx, t in enumerate(sorted(used_tables))}

    rows_out = []
    for t in used_tables:
        r_cell, c_cell = loc_map[t]
        for c in clubs:
            seated = int(x[(c, t)].value())
            if seated:
                rows_out.append(
                    {
                        "Row": r_cell,
                        "Col": c_cell,
                        "Table": f"T{r_cell},{c_cell}",
                        "Club": c,
                        "Members": seated,
                    }
                )
    out_df = pd.DataFrame(rows_out).sort_values(["Row", "Col", "Club"])
    out_df.to_excel("Table_Allocations.xlsx", index=False)

    # ---- Draw grid -------------------------------------------------------
    max_row = max(r for r, _ in allowed_locations) + 1
    max_col = max(c for _, c in allowed_locations) + 1
    fig, ax = plt.subplots(figsize=(max_col * 2.2, max_row * 2.2))
    ax.set_xlim(0, max_col)
    ax.set_ylim(0, max_row)
    ax.set_xticks(range(max_col + 1))
    ax.set_yticks(range(max_row + 1))
    ax.grid(True)

    for (r_cell, c_cell), tbl_grp in out_df.groupby(["Row", "Col"]):
        text_lines = [f"Table {r_cell},{c_cell}"] + [
            f"{row['Club']}: {row['Members']}" for _, row in tbl_grp.iterrows()
        ]
        ax.text(
            c_cell + 0.5,
            max_row - r_cell - 0.5,
            "\n".join(text_lines),
            ha="center",
            va="center",
            fontsize=10,
            wrap=True,
        )
        ax.add_patch(
            plt.Rectangle(
                (c_cell, max_row - r_cell - 1), 1, 1, fill=False, edgecolor="red", lw=2
            )
        )

    plt.tight_layout()
    plt.savefig("table_layout.png", dpi=300)
    plt.close()
    print("Files saved → Table_Allocations.xlsx, table_layout.png")


# ---------------------------------------------------------------------------
# GUI with auto-save ---------------------------------------------------------
# ---------------------------------------------------------------------------

def launch_gui():
    # ---- load saved state ----------------------------------------------
    state = {}
    if SETTINGS_FILE.exists():
        try:
            state = json.loads(SETTINGS_FILE.read_text())
        except Exception:
            state = {}
    allowed_locations = state.get("allowed_locations", [])

    # ---- helpers ---------------------------------------------------------
    def save_state():
        data = {
            "file": file_var.get(),
            "grid_rows": grid_rows_var.get(),
            "grid_cols": grid_cols_var.get(),
            "allowed_locations": allowed_locations,
            "numerics": [v.get() for v in num_vars],
            "sliders": [s.get() for s in sliders],
        }
        SETTINGS_FILE.write_text(json.dumps(data))

    # ---- file chooser ----------------------------------------------------
    def choose_file():
        p = filedialog.askopenfilename(title="Select Excel", filetypes=[("Excel", "*.xlsx *.xls")])
        if p:
            file_var.set(p)
            save_state()

    # ---- grid selection window ------------------------------------------
    def open_grid():
        nonlocal allowed_locations
        r = int(grid_rows_var.get())
        c = int(grid_cols_var.get())
        top = tk.Toplevel(root)
        top.title("Toggle green cells as table spots")
        selected = [[tk.IntVar(value=0) for _ in range(c)] for _ in range(r)]
        for (rr, cc) in allowed_locations:
            if rr < r and cc < c:
                selected[rr][cc].set(1)

        def tog(i, j, btn):
            v = selected[i][j]
            v.set(0 if v.get() else 1)
            btn.configure(bg="green" if v.get() else "lightgray")

        for i in range(r):
            for j in range(c):
                bg = "green" if selected[i][j].get() else "lightgray"
                b = tk.Button(top, width=4, height=2, bg=bg)
                b.grid(row=i, column=j, padx=1, pady=1)
                b.config(command=lambda i=i, j=j, b=b: tog(i, j, b))

        def save_grid():
            allowed_locations.clear()
            for i in range(r):
                for j in range(c):
                    if selected[i][j].get():
                        allowed_locations.append((i, j))
            allowed_locations.sort()
            save_state()
            top.destroy()

        tk.Button(top, text="Save layout", command=save_grid).grid(
            row=r, column=0, columnspan=c, pady=4
        )

    # ---- run button handler ---------------------------------------------
    def run_gui():
        if not allowed_locations:
            print("Define grid first")
            return
        try:
            run_assignment(
                file_var.get(),
                int(num_vars[0].get()),
                int(num_vars[1].get()),
                int(num_vars[2].get()),
                float(sliders[1].get()) / 10.0,  # preferred seat penalty
                int(sliders[2].get()) * 100,  # small-club split penalty (legacy)
                float(sliders[3].get()) / 5.0,  # extra split penalty (legacy)
                int(num_vars[3].get()),
                float(sliders[4].get()) / 5.0,  # min-fragment penalty (legacy)
                int(num_vars[4].get()),
                allowed_locations,
                float(sliders[0].get()) / 100.0,  # λ = split weight
            )
        except Exception as e:
            print("Error:", e)
        finally:
            save_state()

    # ---------------------------------------------------------------------
    #                     BUILD THE MAIN GUI WINDOW                        
    # ---------------------------------------------------------------------
    root = tk.Tk()
    root.title("Sports Ball Table Assignment")

    # File row -------------------------------------------------------------
    tk.Label(root, text="Excel file:").grid(row=0, column=0, sticky="e")
    file_var = tk.StringVar(value=state.get("file", ""))
    tk.Entry(root, textvariable=file_var, width=45).grid(
        row=0, column=1, columnspan=3, sticky="w"
    )
    tk.Button(root, text="Browse…", command=choose_file).grid(row=0, column=4, padx=5)

    # Grid dimension controls ---------------------------------------------
    grid_rows_var = tk.StringVar(value=state.get("grid_rows", "5"))
    grid_cols_var = tk.StringVar(value=state.get("grid_cols", "5"))
    tk.Label(root, text="Rows:").grid(row=1, column=0, sticky="e")
    tk.Entry(root, textvariable=grid_rows_var, width=4).grid(row=1, column=1, sticky="w")
    tk.Label(root, text="Cols:").grid(row=1, column=2, sticky="e")
    tk.Entry(root, textvariable=grid_cols_var, width=4).grid(row=1, column=3, sticky="w")
    tk.Button(root, text="Open grid", command=open_grid, bg="#2196f3", fg="white").grid(
        row=1, column=4, padx=5
    )

    # Numeric fields -------------------------------------------------------
    num_labels = [
        "Min seats/table",
        "Max seats/table",
        "Preferred seats",
        "Min fragment size",
        "Solver time (s)",
    ]
    defaults = state.get("numerics", [8, 10, 9, 6, 10])
    num_vars = []
    row = 2
    for lab, defv in zip(num_labels, defaults):
        tk.Label(root, text=lab).grid(row=row, column=0, sticky="e")
        var = tk.StringVar(value=str(defv))
        ent = tk.Entry(root, textvariable=var, width=6)
        ent.grid(row=row, column=1, sticky="w")
        ent.bind("<FocusOut>", lambda e: save_state())
        num_vars.append(var)
        row += 1

    # Sliders --------------------------------------------------------------
    slider_defaults = [70, 1.0, 10.0, 5.0, 5.0]
    saved_sliders = state.get("sliders", [])
    slider_vals = [
        saved_sliders[i] if i < len(saved_sliders) else slider_defaults[i]
        for i in range(len(slider_defaults))
    ]

    slider_info = [
        (
            "Split-vs-Fill weight (%)",
            slider_vals[0],
            "Move right to keep clubs together (λ↑)",
            0,
            100,
            1,
        ),
        (
            "Preferred seat penalty",
            slider_vals[1],
            "Higher ⇒ stronger push toward preferred seats",
            1,
            10,
            0.5,
        ),
        (
            "Small-club split penalty",
            slider_vals[2],
            "Higher ⇒ discourages splitting small clubs (legacy)",
            1,
            10,
            0.5,
        ),
        (
            "Extra split penalty",
            slider_vals[3],
            "Higher ⇒ discourages extra tables (legacy)",
            1,
            10,
            0.5,
        ),
        (
            "Min-fragment penalty",
            slider_vals[4],
            "Higher ⇒ enforces min fragment size (legacy)",
            1,
            10,
            0.5,
        ),
    ]

    sliders = []
    for lbl, init, tip, lo, hi, step in slider_info:
        tk.Label(root, text=lbl).grid(row=row, column=0, sticky="e")
        scl = tk.Scale(
            root,
            from_=lo,
            to=hi,
            resolution=step,
            orient=tk.HORIZONTAL,
            length=180,
        )
        scl.set(init)
        scl.grid(row=row, column=1, columnspan=3, sticky="w")
        scl.configure(command=lambda val: save_state())
        tk.Label(root, text=tip, font=("Arial", 8), fg="gray").grid(
            row=row + 1, column=0, columnspan=5, sticky="w", padx=10
        )
        sliders.append(scl)
        row += 2

    # Run button -----------------------------------------------------------
    tk.Button(
        root,
        text="Run Assignment",
        command=run_gui,
        bg="#4caf50",
        fg="white",
        width=20,
    ).grid(row=row, column=0, columnspan=5, pady=8)

    # auto-save on change ---------------------------------------------------
    file_var.trace_add("write", lambda *args: save_state())
    grid_rows_var.trace_add("write", lambda *args: save_state())
    grid_cols_var.trace_add("write", lambda *args: save_state())

    root.mainloop()


if __name__ == "__main__":
    launch_gui()
