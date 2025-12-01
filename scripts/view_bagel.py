import os
import re
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

def extract_id_from_name(filename: str) -> str | None:
    """
    Extract the LAST group of digits from the filename.
    Example:
        'model_3_seed1.pae' -> '1'
        'AF_model_005_chimeric.plddt' -> '005'
    """
    nums = re.findall(r"\d+", filename)
    return nums[-1] if nums else None

def load_pae(path: str) -> np.ndarray:
    """Load a PAE matrix from a .pae text file."""
    return np.loadtxt(path, comments="#")

def load_plddt(path: str) -> np.ndarray:
    """Load pLDDT values from a .plddt text file."""
    arr = np.loadtxt(path, comments="#")
    return arr.ravel()

def compute_metrics(plddt: np.ndarray, pae: np.ndarray) -> dict:
    """Compute summary metrics from pLDDT + PAE data."""
    metrics = {}

    metrics["mean_plddt"] = float(np.mean(plddt))
    metrics["median_plddt"] = float(np.median(plddt))
    metrics["frac_plddt_ge_70"] = float(np.mean(plddt >= 70))
    metrics["frac_plddt_ge_90"] = float(np.mean(plddt >= 90))

    metrics["mean_pae"] = float(np.mean(pae))
    metrics["median_pae"] = float(np.median(pae))

    # Composite score (higher = better)
    metrics["composite_score"] = metrics["mean_plddt"] - metrics["mean_pae"]

    return metrics

def save_pae_image(pae: np.ndarray, outpath: str):
    """Generate and save a JPG heatmap for a PAE matrix."""
    plt.figure(figsize=(6, 5))
    plt.imshow(pae, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="PAE (Å)")
    plt.title("Predicted Aligned Error")
    plt.xlabel("Residue index")
    plt.ylabel("Residue index")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    print("\n=== AlphaFold PAE / pLDDT evaluator & ranker ===")
    print("Enter the path to a folder containing your .pae and .plddt files:")
    folder = input("> ").strip()
    try: 
        run_date_time = str(folder.split('/')[-3].split('_')[-2]) + "_" + str(folder.split('/')[-3].split('_')[-1])
    except Exception:
        print("\n⚠️ Warning: Could not extract run date/time from folder name. Using 'results' as default. (Possibly wrong folder structure or path?)\n")
        run_date_time = "results"
    
    if not os.path.isdir(folder):
        print(f"\n❌ Error: '{folder}' is not a valid directory.")
        return

    # Collect files
    pae_files = sorted(glob.glob(os.path.join(folder, "*.pae")))
    plddt_files = sorted(glob.glob(os.path.join(folder, "*.plddt")))

    print(f"\n📁 Found {len(pae_files)} .pae files and {len(plddt_files)} .plddt files.\n")

    # === Create results folder structure ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(
        os.path.join(script_dir, "..", "results", f"results_{run_date_time}"))
    os.makedirs(results_dir, exist_ok=True)

    pae_img_dir = os.path.join(results_dir, "pae_jpgs" + "_" + run_date_time)
    os.makedirs(pae_img_dir, exist_ok=True)

    print(f"[INFO] Saving results to: {results_dir}")
    print(f"[INFO] Saving PAE heatmaps to: {pae_img_dir}\n")

    # === Map files by structure ID ===
    pae_by_id = {}
    plddt_by_id = {}

    for path in pae_files:
        sid = extract_id_from_name(os.path.basename(path))
        if sid:
            pae_by_id.setdefault(sid, path)

    for path in plddt_files:
        sid = extract_id_from_name(os.path.basename(path))
        if sid:
            plddt_by_id.setdefault(sid, path)

    common_ids = sorted(set(pae_by_id) & set(plddt_by_id))

    if not common_ids:
        print("⚠️ No matching IDs between .pae and .plddt files.")
        return

    print(f"🔗 Matched {len(common_ids)} paired structures.\n")

    results = []

    for sid in common_ids:
        pae_path = pae_by_id[sid]
        plddt_path = plddt_by_id[sid]

        print(f"➡️ Evaluating ID {sid}...")

        try:
            pae = load_pae(pae_path)
            plddt = load_plddt(plddt_path)
        except Exception as e:
            print(f"   ❌ Error loading data for ID {sid}: {e}")
            continue

        # Save PAE heatmap
        out_img = os.path.join(pae_img_dir, f"pae_{sid}.jpg")
        save_pae_image(pae, out_img)

        metrics = compute_metrics(plddt, pae)
        metrics["id"] = sid
        metrics["pae_file"] = os.path.basename(pae_path)
        metrics["plddt_file"] = os.path.basename(plddt_path)
        results.append(metrics)

    # === Ranking ===
    results.sort(key=lambda m: m["composite_score"], reverse=True)

    # === Print summary ===
    print("\n===== Ranked models (best → worst) =====\n")
    header = (
        f"{'Rank':>4} {'ID':>5} {'Mean pLDDT':>12} {'Mean PAE':>9} "
        f"{'Frac≥70':>8} {'Frac≥90':>8} {'Score':>8}"
    )
    print(header)
    print("-" * len(header))

    for i, m in enumerate(results, start=1):
        print(
            f"{i:>4} {m['id']:>5} "
            f"{m['mean_plddt']:12.2f} {m['mean_pae']:9.2f} "
            f"{m['frac_plddt_ge_70']:8.2f} {m['frac_plddt_ge_90']:8.2f} "
            f"{m['composite_score']:8.2f}"
        )

    # === Save CSV ===
    csv_path = os.path.join(results_dir, "structure_quality_summary" + "_" + run_date_time + ".csv")
    fieldnames = [
        "id", "pae_file", "plddt_file",
        "mean_plddt", "median_plddt",
        "frac_plddt_ge_70", "frac_plddt_ge_90",
        "mean_pae", "median_pae", "composite_score"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in results:
            writer.writerow(m)

    print(f"\n📊 Summary saved to: {csv_path}")
    print(f"🖼️ PAE heatmaps saved to: {pae_img_dir}\n")
    print("🎉 Done!")

if __name__ == "__main__":
    main()