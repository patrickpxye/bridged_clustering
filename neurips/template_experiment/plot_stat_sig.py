# Recreate GNN vs BKM figure with box color labels in the first subplot
fig, axes = plt.subplots(1, len(cluster_values), figsize=(6 * len(cluster_values), 5), sharey=False)

for idx, cluster in enumerate(cluster_values):
    ax = axes[idx]
    positions = []
    gnn_data, bkm_data = [], []
    pos = 1

    for sup in supervision_levels:
        key = (cluster, sup)
        gnn_vals = results[key]["GNN"]
        bkm_vals = results[key]["BKM"]
        gnn_data.append(gnn_vals)
        bkm_data.append(bkm_vals)
        positions.extend([pos, pos + 1])
        pos += 3

    all_data = [v for pair in zip(bkm_data, gnn_data) for v in pair]
    all_positions = positions

    # Boxplot
    bplot = ax.boxplot(all_data, positions=all_positions, widths=0.6, patch_artist=True)
    for i, patch in enumerate(bplot['boxes']):
        patch.set_facecolor("#1f77b4" if i % 2 == 0 else "#9467bd")

    # Significance annotations
    for i, sup in enumerate(supervision_levels):
        res = next((r for r in gnn_vs_bkm_augmented if r['k'] == cluster and r['sup'] == sup), None)
        if res and "ns" not in res['label']:
            label = res['label']
            y = max(max(bkm_data[i]), max(gnn_data[i])) + 5
            x1, x2 = positions[2 * i], positions[2 * i + 1]
            ax.plot([x1, x1, x2, x2], [y, y + 5, y + 5, y], lw=1.5, c='k')
            ax.text((x1 + x2) / 2, y + 6, label, ha='center', va='bottom', fontsize=12, color='red')

    ax.set_title(f"k = {cluster}", fontsize=14)
    ax.set_xticks([np.mean(positions[i*2:i*2+2]) for i in range(len(supervision_levels))])
    ax.set_xticklabels([str(s) for s in supervision_levels])
    ax.set_xlabel("Supervised Samples per Cluster")
    ax.grid(True, axis='y')
    if idx == 0:
        ax.set_ylabel("MAE")
        # Add custom legend for colors
        ax.legend(handles=[
            plt.Line2D([0], [0], color="#1f77b4", lw=10, label="BKM"),
            plt.Line2D([0], [0], color="#9467bd", lw=10, label="GNN")
        ], loc='upper right', frameon=False)

fig.suptitle("GNN vs BKM: MAE Across Cluster Sizes with Significance Annotations", fontsize=16)
plt.figtext(0.5, -0.02,
    "Significance (paired t-test, N=10 trials per setting):  † : p < 0.1   * : p < 0.05   ** : p < 0.01   *** : p < 0.001",
    ha="center", fontsize=12, color="red")
fig.tight_layout(rect=[0, 0.05, 1, 0.95], w_pad=4)

output_path_labeled = "/mnt/data/gnn_vs_bkm_boxplots_with_legend_no_ns.pdf"
plt.savefig(output_path_labeled, bbox_inches="tight")
plt.show()
