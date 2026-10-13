import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('figures', exist_ok=True)

# Set high-quality publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#dddddd'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.6


# ==============================================================================
# FIGURE 1: Conceptual Alignment Architecture (4 panels)
# ==============================================================================
def create_conceptual_figure():
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    plt.subplots_adjust(wspace=0.15, hspace=0.22)
    
    c_user = '#2b5c8f'
    c_item = '#d95f02'
    c_feat = '#2ca02c'
    c_agg = '#7570b3'
    c_box = '#f7f7f7'
    c_border = '#666666'
    
    # Panel (a): Horizontal Federated Learning (HFL)
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(a) Horizontal Federated Learning (HFL)\nDisjoint Users, Shared Items: Dense FedAvg', fontsize=12, fontweight='bold', pad=10)
    
    srv = patches.FancyBboxPatch((3.0, 7.8), 4.0, 1.8, boxstyle="round,pad=0.1", fc='#e8eaf6', ec=c_agg, lw=1.8)
    ax.add_patch(srv)
    ax.text(5.0, 9.0, 'Central Coordinating Server', ha='center', va='center', fontsize=10, fontweight='bold', color=c_agg)
    ax.text(5.0, 8.3, 'Global Model: $\\mathbf{w}_{t+1} = \\sum_{k=1}^K p_k \\mathbf{w}_{t+1}^{(k)}$\nShared Item Embeddings $\\mathbf{Q}$ (Dense Updates)', ha='center', va='center', fontsize=8.5)
    
    c1 = patches.FancyBboxPatch((0.5, 1.0), 4.0, 5.0, boxstyle="round,pad=0.1", fc=c_box, ec=c_border, lw=1.2)
    ax.add_patch(c1)
    ax.text(2.5, 5.6, 'Client 1 ($k=1$)', ha='center', va='center', fontsize=10, fontweight='bold')
    d1 = patches.Rectangle((1.0, 4.0), 3.0, 1.2, fc='#bbdefb', ec=c_user, lw=1)
    ax.add_patch(d1)
    ax.text(2.5, 4.6, 'Local User Partition $\\mathcal{U}_1$\n(Disjoint: $\\mathcal{U}_1 \\cap \\mathcal{U}_2 = \\emptyset$)', ha='center', va='center', fontsize=8.5, color='#0d47a1')
    i1 = patches.Rectangle((1.0, 2.4), 3.0, 1.2, fc='#ffe0b2', ec=c_item, lw=1)
    ax.add_patch(i1)
    ax.text(2.5, 3.0, 'Shared Item Set $\\mathcal{I}$\n(Full Catalogue Active)', ha='center', va='center', fontsize=8.5, color='#e65100')
    ax.text(2.5, 1.6, 'Local MLP: $\\widehat{y}_{ui} = v^\\top h_2 + b_0$', ha='center', va='center', fontsize=8.5, style='italic')

    c2 = patches.FancyBboxPatch((5.5, 1.0), 4.0, 5.0, boxstyle="round,pad=0.1", fc=c_box, ec=c_border, lw=1.2)
    ax.add_patch(c2)
    ax.text(7.5, 5.6, 'Client 2 ($k=2$)', ha='center', va='center', fontsize=10, fontweight='bold')
    d2 = patches.Rectangle((6.0, 4.0), 3.0, 1.2, fc='#bbdefb', ec=c_user, lw=1)
    ax.add_patch(d2)
    ax.text(7.5, 4.6, 'Local User Partition $\\mathcal{U}_2$\n(Disjoint: $\\mathcal{U}_1 \\cap \\mathcal{U}_2 = \\emptyset$)', ha='center', va='center', fontsize=8.5, color='#0d47a1')
    i2 = patches.Rectangle((6.0, 2.4), 3.0, 1.2, fc='#ffe0b2', ec=c_item, lw=1)
    ax.add_patch(i2)
    ax.text(7.5, 3.0, 'Shared Item Set $\\mathcal{I}$\n(Full Catalogue Active)', ha='center', va='center', fontsize=8.5, color='#e65100')
    ax.text(7.5, 1.6, 'Local MLP: $\\widehat{y}_{ui} = v^\\top h_2 + b_0$', ha='center', va='center', fontsize=8.5, style='italic')

    ax.annotate('', xy=(4.2, 7.8), xytext=(2.5, 6.0), arrowprops=dict(arrowstyle="<->", color=c_agg, lw=1.8))
    ax.annotate('', xy=(5.8, 7.8), xytext=(7.5, 6.0), arrowprops=dict(arrowstyle="<->", color=c_agg, lw=1.8))
    ax.text(3.1, 7.0, 'Full $\\Delta \\mathbf{w}_1$', fontsize=8.5, color=c_agg, fontweight='bold', rotation=35)
    ax.text(6.3, 7.0, 'Full $\\Delta \\mathbf{w}_2$', fontsize=8.5, color=c_agg, fontweight='bold', rotation=-35)

    # Panel (b): Item-Partitioned CF (IP-CF)
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(b) Item-Partitioned CF (IP-CF)\nShared Users, Disjoint Items: Block-Sparse Attenuation', fontsize=12, fontweight='bold', pad=10)
    
    srv = patches.FancyBboxPatch((2.2, 7.8), 5.6, 1.8, boxstyle="round,pad=0.1", fc='#fff3e0', ec='#e65100', lw=1.8)
    ax.add_patch(srv)
    ax.text(5.0, 9.0, 'Global Coordinate Server (Dense FedAvg)', ha='center', va='center', fontsize=10, fontweight='bold', color='#e65100')
    ax.text(5.0, 8.3, 'Block-Sparse Delta: $\\operatorname{supp}(\\Delta \\mathbf{Q}_k) \\subseteq \\mathcal{I}_k$\n$\\Rightarrow$ Item Row Update Attenuation: $\\Delta \\mathbf{q}_i = p_k \\Delta \\mathbf{q}_{i,k} \\approx \\frac{1}{K} \\Delta \\mathbf{q}_{i,k}$', ha='center', va='center', fontsize=8.5, color='#bf360c', fontweight='bold')

    c1 = patches.FancyBboxPatch((0.5, 1.0), 4.0, 5.0, boxstyle="round,pad=0.1", fc=c_box, ec=c_border, lw=1.2)
    ax.add_patch(c1)
    ax.text(2.5, 5.6, 'Client 1 ($k=1$)', ha='center', va='center', fontsize=10, fontweight='bold')
    u1 = patches.Rectangle((1.0, 4.0), 3.0, 1.2, fc='#bbdefb', ec=c_user, lw=1)
    ax.add_patch(u1)
    ax.text(2.5, 4.6, 'Shared Users $\\mathcal{U}$\n(All users observed)', ha='center', va='center', fontsize=8.5, color='#0d47a1')
    p1 = patches.Rectangle((1.0, 2.4), 3.0, 1.2, fc='#ffe0b2', ec=c_item, lw=1)
    ax.add_patch(p1)
    ax.text(2.5, 3.0, 'Disjoint Item Slice $\\mathcal{I}_1$\nActive Rows $\\in \\mathcal{I}_1$', ha='center', va='center', fontsize=8.5, color='#e65100', fontweight='bold')
    ax.text(2.5, 1.6, 'Inactive Rows: $\\Delta \\mathbf{q}_{i \\notin \\mathcal{I}_1} = \\mathbf{0}$', ha='center', va='center', fontsize=8, color='#c62828')

    c2 = patches.FancyBboxPatch((5.5, 1.0), 4.0, 5.0, boxstyle="round,pad=0.1", fc=c_box, ec=c_border, lw=1.2)
    ax.add_patch(c2)
    ax.text(7.5, 5.6, 'Client 2 ($k=2$)', ha='center', va='center', fontsize=10, fontweight='bold')
    u2 = patches.Rectangle((6.0, 4.0), 3.0, 1.2, fc='#bbdefb', ec=c_user, lw=1)
    ax.add_patch(u2)
    ax.text(7.5, 4.6, 'Shared Users $\\mathcal{U}$\n(All users observed)', ha='center', va='center', fontsize=8.5, color='#0d47a1')
    p2 = patches.Rectangle((6.0, 2.4), 3.0, 1.2, fc='#ffe0b2', ec=c_item, lw=1)
    ax.add_patch(p2)
    ax.text(7.5, 3.0, 'Disjoint Item Slice $\\mathcal{I}_2$\nActive Rows $\\in \\mathcal{I}_2$', ha='center', va='center', fontsize=8.5, color='#e65100', fontweight='bold')
    ax.text(7.5, 1.6, 'Inactive Rows: $\\Delta \\mathbf{q}_{i \\notin \\mathcal{I}_2} = \\mathbf{0}$', ha='center', va='center', fontsize=8, color='#c62828')

    ax.annotate('', xy=(4.0, 7.8), xytext=(2.5, 6.0), arrowprops=dict(arrowstyle="<->", color='#e65100', lw=1.8))
    ax.annotate('', xy=(6.0, 7.8), xytext=(7.5, 6.0), arrowprops=dict(arrowstyle="<->", color='#e65100', lw=1.8))
    ax.text(2.4, 7.0, 'Sparse $\\Delta \\mathbf{Q}_1$', fontsize=8.5, color='#e65100', fontweight='bold', rotation=35)
    ax.text(6.7, 7.0, 'Sparse $\\Delta \\mathbf{Q}_2$', fontsize=8.5, color='#e65100', fontweight='bold', rotation=-35)

    # Panel (c): Canonical Vertical Federated Learning (VFL)
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(c) Canonical Vertical Federated Learning (VFL)\nAligned Entities, Disjoint Features: Split Activation Flow', fontsize=12, fontweight='bold', pad=10)

    ent = patches.Rectangle((2.0, 8.4), 6.0, 0.9, fc='#e0f2f1', ec='#00796b', lw=1.5)
    ax.add_patch(ent)
    ax.text(5.0, 8.85, 'Aligned Entity Keys $\\mathcal{E}_1 = \\mathcal{E}_2 = \\mathcal{E}$ (PSI Protocol)', ha='center', va='center', fontsize=9, fontweight='bold', color='#004d40')

    top = patches.FancyBboxPatch((3.0, 5.4), 4.0, 1.8, boxstyle="round,pad=0.1", fc='#f3e5f5', ec='#7b1fa2', lw=1.5)
    ax.add_patch(top)
    ax.text(5.0, 6.4, 'Server / Host Joint Top Model', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#4a148c')
    ax.text(5.0, 5.8, '$\\widehat{y} = g_\\theta([\\mathbf{z}_1; \\mathbf{z}_2])$ & Loss Backprop', ha='center', va='center', fontsize=8.5)

    c1 = patches.FancyBboxPatch((0.5, 0.6), 4.0, 3.8, boxstyle="round,pad=0.1", fc=c_box, ec=c_border, lw=1.2)
    ax.add_patch(c1)
    ax.text(2.5, 3.9, 'Client 1 (Feature Block $\\mathcal{F}_1$)', ha='center', va='center', fontsize=9, fontweight='bold')
    f1 = patches.Rectangle((1.0, 2.3), 3.0, 1.1, fc='#c8e6c9', ec=c_feat, lw=1)
    ax.add_patch(f1)
    ax.text(2.5, 2.85, 'Features: User Demographics\n(Age, Gender, Location)', ha='center', va='center', fontsize=8, color='#1b5e20')
    ax.text(2.5, 1.4, 'Bottom Encoder: $\\mathbf{z}_1 = \\phi_1(\\mathbf{x}_1)$\n(Raw data stays local)', ha='center', va='center', fontsize=8)

    c2 = patches.FancyBboxPatch((5.5, 0.6), 4.0, 3.8, boxstyle="round,pad=0.1", fc=c_box, ec=c_border, lw=1.2)
    ax.add_patch(c2)
    ax.text(7.5, 3.9, 'Client 2 (Feature Block $\\mathcal{F}_2$)', ha='center', va='center', fontsize=9, fontweight='bold')
    f2 = patches.Rectangle((6.0, 2.3), 3.0, 1.1, fc='#c8e6c9', ec=c_feat, lw=1)
    ax.add_patch(f2)
    ax.text(7.5, 2.85, 'Features: Item Content\n(Text TF-IDF, Price, Category)', ha='center', va='center', fontsize=8, color='#1b5e20')
    ax.text(7.5, 1.4, 'Bottom Encoder: $\\mathbf{z}_2 = \\phi_2(\\mathbf{x}_2)$\n(Holds Target $y$ or Split Loss)', ha='center', va='center', fontsize=8)

    ax.annotate('', xy=(3.8, 5.4), xytext=(2.5, 4.4), arrowprops=dict(arrowstyle="->", color='#7b1fa2', lw=1.8))
    ax.annotate('', xy=(6.2, 5.4), xytext=(7.5, 4.4), arrowprops=dict(arrowstyle="->", color='#7b1fa2', lw=1.8))
    ax.text(2.3, 4.9, 'Activations $\\mathbf{z}_1$', fontsize=8, color='#7b1fa2', fontweight='bold')
    ax.text(6.9, 4.9, 'Activations $\\mathbf{z}_2$', fontsize=8, color='#7b1fa2', fontweight='bold')

    # Panel (d): Pseudo-VFL via Zero-Padding (Audited CBF Practice)
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('(d) Pseudo-VFL via Zero-Padding (Empirical CBF Audit)\nFeature Slicing with Padded Zeros under Standard FedAvg', fontsize=12, fontweight='bold', pad=10)

    srv = patches.FancyBboxPatch((2.5, 7.8), 5.0, 1.8, boxstyle="round,pad=0.1", fc='#ffebee', ec='#c62828', lw=1.8)
    ax.add_patch(srv)
    ax.text(5.0, 9.0, 'Standard FedAvg Server (Enforced Uniform Shape)', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#c62828')
    ax.text(5.0, 8.3, 'No Split Activation Exchange!\nDense Averaging of Weights $\\mathbf{w}_{t+1} = \\sum_k p_k \\mathbf{w}_k$', ha='center', va='center', fontsize=8.5)

    c1 = patches.FancyBboxPatch((0.5, 0.6), 4.0, 6.2, boxstyle="round,pad=0.1", fc=c_box, ec=c_border, lw=1.2)
    ax.add_patch(c1)
    ax.text(2.5, 6.3, 'Client 1 (Holds Features $\\mathcal{F}_1$)', ha='center', va='center', fontsize=9, fontweight='bold')
    a1 = patches.Rectangle((1.0, 4.8), 3.0, 1.0, fc='#c8e6c9', ec=c_feat, lw=1)
    ax.add_patch(a1)
    ax.text(2.5, 5.3, 'Active Features $\\mathcal{F}_1$\n(Real Demographics)', ha='center', va='center', fontsize=8, color='#1b5e20')
    z1 = patches.Rectangle((1.0, 3.4), 3.0, 1.0, fc='#ffcdd2', ec='#e53935', lw=1, linestyle='--')
    ax.add_patch(z1)
    ax.text(2.5, 3.9, 'Missing Features $\\mathcal{F}_2 \\leftarrow 0.0$\n(Artificially Padded with Zeros)', ha='center', va='center', fontsize=8, color='#b71c1c', fontweight='bold')
    ax.text(2.5, 2.0, 'Full Network Architecture Trained:\n$\\mathbf{W}_{\\text{in}} \\in \\mathbb{R}^{d \\times (d_1 + d_2)}$\n(Zero-inputs pollute weight updates)', ha='center', va='center', fontsize=7.5)

    c2 = patches.FancyBboxPatch((5.5, 0.6), 4.0, 6.2, boxstyle="round,pad=0.1", fc=c_box, ec=c_border, lw=1.2)
    ax.add_patch(c2)
    ax.text(7.5, 6.3, 'Client 2 (Holds Features $\\mathcal{F}_2$)', ha='center', va='center', fontsize=9, fontweight='bold')
    z2 = patches.Rectangle((6.0, 4.8), 3.0, 1.0, fc='#ffcdd2', ec='#e53935', lw=1, linestyle='--')
    ax.add_patch(z2)
    ax.text(7.5, 5.3, 'Missing Features $\\mathcal{F}_1 \\leftarrow 0.0$\n(Artificially Padded with Zeros)', ha='center', va='center', fontsize=8, color='#b71c1c', fontweight='bold')
    a2 = patches.Rectangle((6.0, 3.4), 3.0, 1.0, fc='#c8e6c9', ec=c_feat, lw=1)
    ax.add_patch(a2)
    ax.text(7.5, 3.9, 'Active Features $\\mathcal{F}_2$\n(Real Content Features)', ha='center', va='center', fontsize=8, color='#1b5e20')
    ax.text(7.5, 2.0, 'Full Network Architecture Trained:\n$\\mathbf{W}_{\\text{in}} \\in \\mathbb{R}^{d \\times (d_1 + d_2)}$\n(Zero-inputs pollute weight updates)', ha='center', va='center', fontsize=7.5)

    ax.annotate('', xy=(4.2, 7.8), xytext=(2.5, 6.8), arrowprops=dict(arrowstyle="<->", color='#c62828', lw=1.8))
    ax.annotate('', xy=(5.8, 7.8), xytext=(7.5, 6.8), arrowprops=dict(arrowstyle="<->", color='#c62828', lw=1.8))
    ax.text(2.6, 7.3, 'Full $\\mathbf{w}_1$', fontsize=8, color='#c62828', fontweight='bold')
    ax.text(6.6, 7.3, 'Full $\\mathbf{w}_2$', fontsize=8, color='#c62828', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/conceptual-alignment.png', dpi=300)
    plt.close()
    print("Created figures/conceptual-alignment.png")


# ==============================================================================
# FIGURE 2: Round Trajectories with Round 9 Learning Rate Step Decay
# ==============================================================================
def create_trajectories_figure():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    rounds = np.arange(1, 11)

    cf_hfl_2 = [2.16, 1.82, 1.55, 1.30, 1.25, 1.20, 1.16, 1.12, 0.90, 0.86]
    cf_ip_2  = [1.62, 1.55, 1.50, 1.48, 1.42, 1.38, 1.33, 1.29, 0.95, 0.91]

    cf_hfl_5 = [2.21, 1.80, 1.50, 1.42, 1.40, 1.35, 1.32, 1.30, 0.99, 0.98]
    cf_ip_5  = [2.27, 1.70, 1.52, 1.48, 1.45, 1.42, 1.38, 1.35, 0.96, 0.94]

    cbf_hfl_2 = [1.64, 1.57, 1.57, 1.56, 1.55, 1.49, 1.50, 1.51, 1.15, 1.20]
    cbf_vl_2  = [1.37, 1.36, 1.33, 1.32, 1.31, 1.32, 1.32, 1.30, 1.18, 1.12]

    cbf_hfl_5 = [2.11, 1.65, 1.58, 1.55, 1.56, 1.52, 1.55, 1.56, 1.24, 1.19]
    cbf_vl_5  = [1.66, 1.50, 1.45, 1.43, 1.40, 1.38, 1.37, 1.36, 1.18, 1.12]

    panels = [
        (axes[0, 0], 'CF, 2 Clients', cf_hfl_2, cf_ip_2, 'HFL', 'IP-CF'),
        (axes[0, 1], 'CF, 5 Clients', cf_hfl_5, cf_ip_5, 'HFL', 'IP-CF'),
        (axes[1, 0], 'CBF, 2 Clients', cbf_hfl_2, cbf_vl_2, 'HFL (Macro)', 'VL-CBF (Joint)'),
        (axes[1, 1], 'CBF, 5 Clients', cbf_hfl_5, cbf_vl_5, 'HFL (Macro)', 'VL-CBF (Joint)')
    ]

    for ax, title, s1, s2, l1, l2 in panels:
        ax.plot(rounds, s1, marker='o', color='#1f77b4', lw=2.0, label=l1)
        ax.plot(rounds, s2, marker='s', color='#ff7f0e', lw=2.0, ls='--', label=l2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Server Round', fontsize=10)
        ax.set_ylabel('Reported Round-Level MSE', fontsize=10)
        ax.set_xticks(rounds)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9.5, loc='upper right')

        ax.axvspan(8.5, 10.5, color='#ffffcc', alpha=0.35, zorder=0)
        ax.annotate('LR Step Decay\n($\\eta \\to 0.2\\eta$)', 
                    xy=(9, min(s1[8], s2[8])), xytext=(6.5, min(s1[8], s2[8]) + 0.35),
                    arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.2, headwidth=6),
                    fontsize=8.5, fontweight='bold', color='#b71c1c',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728", lw=0.8))

    plt.tight_layout()
    plt.savefig('figures/round-trajectories.png', dpi=300)
    plt.close()
    print("Created figures/round-trajectories.png")


# ==============================================================================
# FIGURE 3: Final MSE Point Estimates (Evaluation Units Separated)
# ==============================================================================
def create_final_mse_figure():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), dpi=300, sharey=True)

    # Panel 1: CF (Macro)
    ax = axes[0]
    x = np.array([0, 1])
    width = 0.35
    hfl_cf = [1.0076, 1.0458]
    ip_cf  = [1.0122, 1.0521]
    
    b1 = ax.bar(x - width/2, hfl_cf, width, label='HFL', color='#1f77b4', edgecolor='black', lw=0.8)
    b2 = ax.bar(x + width/2, ip_cf, width, label='IP-CF', color='#ff7f0e', hatch='//', edgecolor='black', lw=0.8)
    ax.set_title('CF (Macro Evaluation)', fontsize=11.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['2 Clients', '5 Clients'], fontsize=10)
    ax.set_ylabel('Final Preference-Score MSE', fontsize=11)
    ax.set_ylim(0, 1.35)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9.5)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f'{b.get_height():.4f}', ha='center', va='bottom', fontsize=8.5)

    # Panel 2: CBF HFL (Macro)
    ax = axes[1]
    hfl_cbf = [1.0675, 1.0772]
    b3 = ax.bar(x, hfl_cbf, width*1.2, color='#aec7e8', edgecolor='black', lw=0.8)
    ax.set_title('CBF: HFL (Macro Evaluation)', fontsize=11.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['2 Clients', '5 Clients'], fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for b in b3:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f'{b.get_height():.4f}', ha='center', va='bottom', fontsize=8.5)

    # Panel 3: CBF VL-CBF (Joint)
    ax = axes[2]
    vl_cbf = [1.0490, 1.0500]
    b4 = ax.bar(x, vl_cbf, width*1.2, color='#c7c7c7', hatch='\\\\', edgecolor='black', lw=0.8)
    ax.set_title('CBF: VL-CBF (Joint Evaluation)', fontsize=11.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['2 Clients', '5 Clients'], fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for b in b4:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f'{b.get_height():.4f}', ha='center', va='bottom', fontsize=8.5)

    plt.tight_layout()
    plt.savefig('figures/final-mse.png', dpi=300)
    plt.close()
    print("Created figures/final-mse.png")


# ==============================================================================
# FIGURE 4: MovieLens 100K Multi-Seed Benchmark across K in {2, 5, 10}
# ==============================================================================
def create_movielens_benchmark_figure():
    json_path = 'experiments/benchmark_results.json'
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} does not exist yet. Skipping benchmark figure.")
        return

    with open(json_path) as f:
        res = json.load(f)

    # Check available Ks
    available_ks = [2, 5]
    if "HFL-10" in res:
        available_ks = [2, 5, 10]

    n_ks = len(available_ks)
    x = np.arange(n_ks)
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), dpi=300)

    # Panel (a): Macro MSE
    ax = axes[0]
    hfl_mse = [res[f'HFL-{k}']['summary']['mse']['macro_mean']['mean'] for k in available_ks]
    hfl_mse_sd = [res[f'HFL-{k}']['summary']['mse']['macro_mean']['sd'] for k in available_ks]

    ip_mse = [res[f'IP-CF-{k}']['summary']['mse']['macro_mean']['mean'] for k in available_ks]
    ip_mse_sd = [res[f'IP-CF-{k}']['summary']['mse']['macro_mean']['sd'] for k in available_ks]

    has_ac = f'AC-IP-CF-{available_ks[0]}' in res
    if has_ac:
        ac_mse = [res[f'AC-IP-CF-{k}']['summary']['mse']['macro_mean']['mean'] for k in available_ks]
        ac_mse_sd = [res[f'AC-IP-CF-{k}']['summary']['mse']['macro_mean']['sd'] for k in available_ks]

        ax.bar(x - width, hfl_mse, width, yerr=hfl_mse_sd, capsize=3, label='HFL', color='#1f77b4', edgecolor='black', lw=0.8)
        ax.bar(x, ip_mse, width, yerr=ip_mse_sd, capsize=3, label='IP-CF (Dense FedAvg)', color='#ff7f0e', hatch='//', edgecolor='black', lw=0.8)
        ax.bar(x + width, ac_mse, width, yerr=ac_mse_sd, capsize=3, label='AC-IP-CF (Renorm)', color='#2ca02c', edgecolor='black', lw=0.8)
    else:
        ax.bar(x - width/2, hfl_mse, width, yerr=hfl_mse_sd, capsize=3, label='HFL', color='#1f77b4', edgecolor='black', lw=0.8)
        ax.bar(x + width/2, ip_mse, width, yerr=ip_mse_sd, capsize=3, label='IP-CF (Dense)', color='#ff7f0e', hatch='//', edgecolor='black', lw=0.8)

    ax.set_title('(a) Prediction Error (Macro MSE)\nMovieLens 100K ($n=5$ seeds)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k} Clients' for k in available_ks], fontsize=10)
    ax.set_ylabel('Macro MSE (Lower is Better)', fontsize=10.5)
    ax.set_ylim(0.80, 0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='upper right')

    # Panel (b): Ranking Quality (Recall@10)
    ax = axes[1]
    hfl_rec = [res[f'HFL-{k}']['summary']['recall@10']['macro_mean']['mean'] for k in available_ks]
    hfl_rec_sd = [res[f'HFL-{k}']['summary']['recall@10']['macro_mean']['sd'] for k in available_ks]

    ip_rec = [res[f'IP-CF-{k}']['summary']['recall@10']['macro_mean']['mean'] for k in available_ks]
    ip_rec_sd = [res[f'IP-CF-{k}']['summary']['recall@10']['macro_mean']['sd'] for k in available_ks]

    if has_ac:
        ac_rec = [res[f'AC-IP-CF-{k}']['summary']['recall@10']['macro_mean']['mean'] for k in available_ks]
        ac_rec_sd = [res[f'AC-IP-CF-{k}']['summary']['recall@10']['macro_mean']['sd'] for k in available_ks]

        ax.bar(x - width, hfl_rec, width, yerr=hfl_rec_sd, capsize=3, label='HFL', color='#1f77b4', edgecolor='black', lw=0.8)
        ax.bar(x, ip_rec, width, yerr=ip_rec_sd, capsize=3, label='IP-CF (Dense FedAvg)', color='#ff7f0e', hatch='//', edgecolor='black', lw=0.8)
        ax.bar(x + width, ac_rec, width, yerr=ac_rec_sd, capsize=3, label='AC-IP-CF (Renorm)', color='#2ca02c', edgecolor='black', lw=0.8)
    else:
        ax.bar(x - width/2, hfl_rec, width, yerr=hfl_rec_sd, capsize=3, label='HFL', color='#1f77b4', edgecolor='black', lw=0.8)
        ax.bar(x + width/2, ip_rec, width, yerr=ip_rec_sd, capsize=3, label='IP-CF (Dense)', color='#ff7f0e', hatch='//', edgecolor='black', lw=0.8)

    ax.set_title('(b) Ranking Quality (Recall@10)\nDemonstrating $1/K$ Attenuation & Recovery', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k} Clients' for k in available_ks], fontsize=10)
    ax.set_ylabel('Recall@10 (Higher is Better)', fontsize=10.5)
    ax.set_ylim(0.20, 0.52)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='lower left')

    # Panel (c): Between-Client Dispersion (Recall@10 CV)
    ax = axes[2]
    hfl_cv = [res[f'HFL-{k}']['summary']['recall@10']['between_client_cv']['mean'] for k in available_ks]
    ip_cv  = [res[f'IP-CF-{k}']['summary']['recall@10']['between_client_cv']['mean'] for k in available_ks]

    if has_ac:
        ac_cv = [res[f'AC-IP-CF-{k}']['summary']['recall@10']['between_client_cv']['mean'] for k in available_ks]

        b1 = ax.bar(x - width, hfl_cv, width, label='HFL', color='#1f77b4', edgecolor='black', lw=0.8)
        b2 = ax.bar(x, ip_cv, width, label='IP-CF (Dense FedAvg)', color='#ff7f0e', hatch='//', edgecolor='black', lw=0.8)
        b3 = ax.bar(x + width, ac_cv, width, label='AC-IP-CF (Renorm)', color='#2ca02c', edgecolor='black', lw=0.8)
        all_bars = list(b1) + list(b2) + list(b3)
    else:
        b1 = ax.bar(x - width/2, hfl_cv, width, label='HFL', color='#1f77b4', edgecolor='black', lw=0.8)
        b2 = ax.bar(x + width/2, ip_cv, width, label='IP-CF (Dense)', color='#ff7f0e', hatch='//', edgecolor='black', lw=0.8)
        all_bars = list(b1) + list(b2)

    ax.set_title('(c) Between-Client Dispersion (CV)\nRanking Variance across Federation Clients', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k} Clients' for k in available_ks], fontsize=10)
    ax.set_ylabel('Recall@10 Client CV ($s/\\mu$)', fontsize=10.5)
    ax.set_ylim(0, 0.28)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=9, loc='upper left')

    for b in all_bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.005, f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    plt.savefig('figures/movielens-benchmark.png', dpi=300)
    plt.close()
    print("Created figures/movielens-benchmark.png")


if __name__ == '__main__':
    create_conceptual_figure()
    create_trajectories_figure()
    create_final_mse_figure()
    create_movielens_benchmark_figure()
