#!/usr/bin/env python3
"""
Generate a PCA-based Bagel landscape.

Coordinates:
    x = PC1 (PCA on state-level features)
    y = PC2
    z = "landscape height"
        - By default: a composite quality score similar to load_pae.py:
          quality_score = state_A:global_pLDDT - state_A:cross_PAE
        - Falls back to an energy column if composite can't be computed.

Outputs go into:
    ./results/
        pca_landscape.csv
        energy_landscape_pca.png
        energy_landscape_pca_trajectory.gif (if a trajectory is available)
"""

import argparse
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# -------------------- CONFIG --------------------

FEATURE_COLUMNS = [
    "state_A:pTM",
    "state_A:global_pLDDT",
    "state_A:hydrophobic",
    "state_A:cross_PAE",
    "state_A:separation",
]

ENERGY_COLUMN_CANDIDATES = [
    "system_energy",
    "system:energy",
    "energy",
    "total_energy",
    "state_A:energy",
]

STEP_COLUMN_CANDIDATES = ["step", "iteration", "sample", "index"]


# -------------------- HELPERS --------------------

def find_energies_csv(sim_root: Path) -> Path:
    """Find energies.csv under simulated_tempering_*"""
    best_path = sim_root / "best" / "energies.csv"
    if best_path.exists():
        return best_path
    cands = list(sim_root.rglob("energies.csv"))
    if not cands:
        raise FileNotFoundError(f"No energies.csv found under {sim_root}")
    return cands[0]


def choose_column(df: pd.DataFrame, candidates):
    """Pick first matching column from a list."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in {list(df.columns)}")


def check_feature_columns(df: pd.DataFrame):
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required feature columns:\n"
            f"  {missing}\n"
            "Available columns:\n"
            f"  {list(df.columns)}\n"
        )


def add_composite_quality_score(df: pd.DataFrame) -> tuple[pd.DataFrame, str, bool]:
    """
    Add a 'quality_score' column analogous to load_pae.py's composite_score
    (mean_plddt - mean_pae).

    Here we approximate:
        quality_score = state_A:global_pLDDT - state_A:cross_PAE

    Returns:
        (df_with_score, col_used_for_landscape, higher_is_better)
    """
    col_plddt = "state_A:global_pLDDT"
    col_pae = "state_A:cross_PAE"

    if col_plddt in df.columns and col_pae in df.columns:
        df = df.copy()
        df["quality_score"] = (
            df[col_plddt].astype(float) - df[col_pae].astype(float)
        )
        print(
            "[INFO] Using composite 'quality_score' as landscape height:\n"
            f"       quality_score = {col_plddt} - {col_pae}\n"
            "       (higher = better, mirroring load_pae.py's composite_score)"
        )
        return df, "quality_score", True

    print(
        "[WARN] Could not compute composite 'quality_score' "
        f"(need {col_plddt} and {col_pae})."
    )
    return df, "", False


def run_pca(df, feature_cols, landscape_col, step_col=None, higher_is_better=False):
    """Standardise → PCA → compute landscape height and a scaled z for plotting."""
    mask = df[feature_cols].notna().all(axis=1)
    df = df.loc[mask].copy()

    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X_scaled)

    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]

    # --- store the raw landscape metric (energy or composite score) ---
    df["landscape_raw"] = df[landscape_col].astype(float)

    # --- map "better" → lower height for the landscape ---
    if higher_is_better:
        # e.g. composite_score: higher is better → invert
        df["energy_for_landscape"] = -df["landscape_raw"]
    else:
        # e.g. physical energy: lower is better → use as-is
        df["energy_for_landscape"] = df["landscape_raw"]

    # --- scale z for nicer visual separation ---
    z = df["energy_for_landscape"].values
    mu = np.mean(z)
    sigma = np.std(z)

    if sigma == 0:
        # all the same → just use raw
        df["z_plot"] = z
    else:
        df["z_plot"] = (z - mu) / sigma  # z-score

    if step_col:
        df["trajectory_step"] = df[step_col]

    print("\n[INFO] Variance explained:")
    for i, r in enumerate(pca.explained_variance_ratio_, 1):
        print(f"  PC{i}: {r:.3f}")

    return df


def plot_landscape(df, out_path: Path,
                   colour_by="landscape_raw",
                   colour_label="Composite score (pLDDT - mean PAE)"):
    """3D scatter + (optional) trajectory line."""
    if colour_by not in df.columns:
        raise KeyError(f"{colour_by} not found in df columns.")

    if "trajectory_step" in df.columns:
        df_plot = df.sort_values("trajectory_step").copy()
    else:
        df_plot = df.copy()

    x = df_plot["PC1"].values
    y = df_plot["PC2"].values
    z = df_plot["z_plot"].values          # <--- use scaled height
    c = df_plot[colour_by].values         # <--- raw composite score / energy

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(x, y, z, c=c, s=22, cmap="viridis")

    # optional trajectory line
    if "trajectory_step" in df_plot.columns:
        ax.plot(x, y, z, linewidth=1, alpha=0.5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("Scaled landscape height (lower = better)")

    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label(colour_label)    # e.g. Composite score (pLDDT - mean PAE)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] Saved: {out_path}")


def animate_trajectory(df, out_path: Path, colour_by="energy_for_landscape", z_label="Height"):
    """
    Create a simple 3D animation of the MCMC trajectory in PCA space.

    The trajectory is ordered by 'trajectory_step'.
    Saves a GIF to out_path (requires Pillow backend installed for matplotlib).
    """
    if "trajectory_step" not in df.columns:
        print("[INFO] No trajectory_step column found; skipping animation.")
        return

    if colour_by not in df.columns:
        print(f"[WARN] {colour_by} not found in df; using 'energy_for_landscape' instead.")
        colour_by = "energy_for_landscape"

    df_anim = df.sort_values("trajectory_step").copy()

    x = df_anim["PC1"].values
    y = df_anim["PC2"].values
    z = df_anim["energy_for_landscape"].values
    c = df_anim[colour_by].values

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # set static axis labels
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel(z_label)

    # determine ranges for nicer consistent framing
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(np.min(z), np.max(z))

    # initialise empty plot objects
    scat = ax.scatter([], [], [], s=30)
    line, = ax.plot([], [], [], linewidth=1.2, alpha=0.6)
    current_point, = ax.plot([], [], [], marker="o", markersize=8, color="red")

    cb = fig.colorbar(scat, ax=ax, shrink=0.7)
    cb.set_label(colour_by)


    def init():
        scat._offsets3d = ([], [], [])
        scat.set_array(np.array([]))
        line.set_data([], [])
        line.set_3d_properties([])
        current_point.set_data([], [])
        current_point.set_3d_properties([])
        return scat, line, current_point

    def update(frame):
        # frame is index along the trajectory
        xi = x[: frame + 1]
        yi = y[: frame + 1]
        zi = z[: frame + 1]
        ci = c[: frame + 1]

        scat._offsets3d = (xi, yi, zi)
        scat.set_array(ci)
        line.set_data(xi, yi)
        line.set_3d_properties(zi)

        current_point.set_data([xi[-1]], [yi[-1]])
        current_point.set_3d_properties([zi[-1]])

        ax.set_title(f"Bagel MCMC Trajectory (step {frame + 1}/{len(x)})")
        
        return scat, line, current_point

    n_frames = len(x)
    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=30,
        blit=False,
    )

    try:
        ani.save(out_path, writer="pillow", fps=20)
        print(f"[INFO] Saved animation: {out_path}")
    except Exception as e:
        print(f"[WARN] Could not save animation ({e}). "
              "Make sure Pillow is installed for GIF support.")

    plt.close(fig)


# -------------------- MAIN --------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("sim_root", nargs="?", help="Path to Bagel simulated_tempering_* directory")
    p.add_argument(
        "--colour-by",
        default="energy_for_landscape",
        help="Column to colour points by (default: energy_for_landscape)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ask interactively if path not given
    if args.sim_root is None:
        root = input("Enter path to simulated_tempering_YYMMDD_HHMMSS: ").strip()
    else:
        root = args.sim_root

    sim_root = Path(root).expanduser().resolve()
    if not sim_root.exists():
        print(f"[ERROR] {sim_root} does not exist.")
        sys.exit(1)

    try: 
        run_date_time = str(root.split('_')[-2]) + "_" + str(root.split('_')[-1])
    except Exception:
        print("\n⚠️ Warning: Could not extract run date/time from folder name. Using 'results' as default. (Possibly wrong folder structure or path?)\n")
        run_date_time = str(np.random.randint(100000,999999))

    print(f"[INFO] Using Bagel output at: {sim_root}")

    # Results folder in CURRENT WORKING DIRECTORY
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(
        os.path.join(script_dir, "..", "results", f"results_{run_date_time}"))
    os.makedirs(results_dir, exist_ok=True)

    # locate and load energies.csv
    energies_path = find_energies_csv(sim_root)
    print(f"[INFO] Using energies.csv: {energies_path}")

    df = pd.read_csv(energies_path)

    # validate columns for PCA
    check_feature_columns(df)

    # pick a raw energy column (for reference / colouring)
    energy_col = choose_column(df, ENERGY_COLUMN_CANDIDATES)
    print(f"[INFO] Raw energy column detected: {energy_col}")

    # detect a trajectory column if present
    try:
        step_col = choose_column(df, STEP_COLUMN_CANDIDATES)
        print(f"[INFO] Trajectory column: {step_col}")
    except KeyError:
        step_col = None
        print("[INFO] No trajectory/step column found.")

    # add composite score in the spirit of load_pae.py
    df, landscape_col, higher_is_better = add_composite_quality_score(df)

    # if we couldn't compute the composite, fall back to the energy column
    if not landscape_col:
        landscape_col = energy_col
        higher_is_better = False  # lower energy = better
        print(
            f"[INFO] Falling back to '{energy_col}' as landscape height "
            "(lower = better)."
        )

    # PCA
    df_pca = run_pca(
        df,
        FEATURE_COLUMNS,
        landscape_col,
        step_col=step_col,
        higher_is_better=higher_is_better,
    )
    
    # mark the best state (for plotting a star)
    best_idx = df_pca["energy_for_landscape"].idxmin()
    if higher_is_better:
        print("[INFO] Best state = max(quality_score) (mapped to lowest height).")
        z_label = "Composite score (pLDDT - cross_PAE, inverted)"
    else:
        print("[INFO] Best state = min(energy).")
        z_label = "Energy"

    df_pca["is_best"] = False
    df_pca.loc[best_idx, "is_best"] = True

    # keep the raw energy around for colouring if desired
    if energy_col in df_pca.columns:
        pass  # already there
    elif energy_col in df.columns:
        # reattach from original df (same index)
        df_pca[energy_col] = df.loc[df_pca.index, energy_col].values

    # save CSV in results/
    pca_csv = results_dir + "/" + "pca_landscape.csv"
    df_pca.to_csv(pca_csv, index=False)
    print(f"[INFO] Saved PCA CSV: {pca_csv}")
    
    # Determine the colour label
    if higher_is_better:
        colour_label = "Composite score (pLDDT – mean PAE)"
    else:
        colour_label = "Energy"

    # static 3D plot
    landscape_png = results_dir + "/" + "energy_landscape_pca.png"
    plot_landscape(
        df_pca,
        landscape_png,
        colour_by="landscape_raw",
        colour_label=colour_label
    )

    # animated trajectory (if possible)
    anim_gif = results_dir + "/" + "energy_landscape_pca_trajectory.gif"
    animate_trajectory(df_pca, anim_gif, colour_by=args.colour_by, z_label=z_label)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()