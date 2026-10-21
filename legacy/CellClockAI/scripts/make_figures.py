"""Generate publication figures directly from saved pilot results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 240, "pdf.fonttype": 42})
COLORS = ["#244b74", "#238a7b", "#a8732b", "#b54747"]


def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(OUT / (name + ".png"), bbox_inches="tight")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.set(xlim=(0, 10.5), ylim=(0, 6.3)); ax.axis("off")
    def box(x, y, w, h, title, lines, color="#e9eff5"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.06,rounding_size=.06",
                                  facecolor=color, edgecolor="#a3b3c2", linewidth=.9))
        ax.text(x+w/2, y+h-.28, title, ha="center", va="center", weight="bold", fontsize=10)
        ax.text(x+w/2, y+h/2-.14, lines, ha="center", va="center", fontsize=9.1, linespacing=1.45)
    def arrow(a, b, label=""):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color="#34485d"))
        if label:ax.text((a[0]+b[0])/2, (a[1]+b[1])/2+.1, label, ha="center", fontsize=8)
    box(.2, 4.2, 3.0, 1.7, "Public reference atlas", "Embryogenesis and adult ageing\nReprogramming time courses\nRNA, methylation, accessibility")
    box(3.7, 4.2, 3.0, 1.7, "Intervention evidence", "Randomized target assignment\nIndependent donors and guides\nTime, exposure and controls", "#e7f2ef")
    box(7.2, 4.2, 3.0, 1.7, "Independent outcomes", "Frozen clocks as secondary readouts\nFunctional recovery and cell identity\nGenome integrity and viability", "#f4eddf")
    box(2.6, 2.0, 5.3, 1.5, "Action conditioned regulatory dynamics", "Masked modality encoders and sparse prior graph\nSeparate functional and identity prediction heads\nUncertainty and intervention support checks")
    box(.3, .1, 3.0, 1.2, "Candidate nomination", "Regulator or cis element\nContext and evidence tier", "#e7f2ef")
    box(3.8, .1, 3.0, 1.2, "Independent confirmation", "Necessity, sufficiency and rescue\nWashout and challenge tests", "#e7f2ef")
    box(7.3, .1, 2.9, 1.2, "Abstain or falsify", "Unsupported combinations\nClock only or identity loss", "#f7e8e8")
    for start in [(1.7,4.2),(5.2,4.2),(8.7,4.2)]:arrow(start,(5.25,3.5))
    for end in [(1.8,1.3),(5.3,1.3),(8.7,1.3)]:arrow((5.25,2),end)
    ax.text(.2,6.18,"Proposed large scale study",weight="bold",fontsize=12)
    save(fig,"figure1_study_architecture")


def public():
    d = pd.read_csv(ROOT / "results/public_predictions.csv")
    d = d.sort_values(["donor", "treated", "duration", "experiment"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios":[1.6,1]})
    ax = axes[0]
    y = np.arange(len(d))
    ax.scatter(y, d.leave_donor_out_probability, c=np.where(d.treated==1,COLORS[1],COLORS[0]),s=27)
    ax.axhline(.5, color="#777",lw=.8,ls="--")
    centers = []
    for donor, group in d.groupby("donor",sort=True):
        positions = np.flatnonzero(d.donor.to_numpy()==donor)
        centers.append((positions.mean(),donor))
        ax.axvline(positions.max()+.5,color="#ddd",lw=.6)
    ax.set_xticks([a for a,b in centers],[b for a,b in centers])
    ax.set(xlabel="Held out donor",ylabel="Predicted treatment probability",ylim=(-.04,1.04),
           title="A  Recovery samples only  n = 34")
    ax.text(.02,.61,"Blue  control\nGreen  reprogrammed",transform=ax.transAxes,fontsize=9)
    null = pd.read_csv(ROOT / "results/public_permutation_null.csv").null_auroc
    axes[1].hist(null,bins=np.linspace(0,1,16),color="#a9b8c8",edgecolor="white")
    axes[1].axvline(1,color=COLORS[1],lw=2)
    axes[1].set(xlabel="Pooled donor holdout AUROC",ylabel="Permutation count",xlim=(0,1.05),
                title="B  Within block label permutations")
    axes[1].text(.04,.9,"Observed  1.00\nPermutation p  0.0099",transform=axes[1].transAxes,fontsize=9)
    fig.tight_layout();save(fig,"figure2_public_pilot")


def synthetic_rmse():
    d = pd.read_csv(ROOT / "results/synthetic_summary.csv",keep_default_na=False)
    scenarios=["additive","synergy","redundant","stress_interaction","unseen_synergy","no_causal_effect"]
    models=["linear_function","interaction_function","neural_function","interaction_clock"]
    labels=["Linear functional","Interaction functional","Neural functional","Clock only"]
    fig, axes = plt.subplots(2,3,figsize=(10.5,6),sharey=True)
    titles=["Additive","Synergy","Redundant","Stress dependent","Unseen synergy","No causal effect"]
    for ax,scenario,title in zip(axes.flat,scenarios,titles):
        sub=d[d.scenario==scenario].set_index("model").loc[models]
        ax.bar(np.arange(4),sub.functional_mean_rmse_mean,yerr=sub.functional_mean_rmse_std,
               color=COLORS,width=.7,capsize=3,error_kw={"lw":.8})
        ax.set_xticks(range(4),["Linear","Interact.","Neural","Clock"],rotation=18)
        ax.set_title(title,fontsize=10);ax.set_ylim(0,1.08)
    for ax in axes[:,0]:ax.set_ylabel("Functional mean RMSE")
    fig.text(.5,.99,"Synthetic benchmark   mean and SD over 20 seeds   arbitrary units",ha="center",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96));save(fig,"figure3_synthetic_error")


def ranking():
    d=pd.read_csv(ROOT / "results/synthetic_rankings.csv")
    scenes=["additive","synergy","redundant","stress_interaction","unseen_synergy"]
    methods=["observational_correlation","linear_function","interaction_function","neural_function","interaction_clock"]
    values=d.groupby(["scenario","method"]).precision_at_2.mean().unstack().loc[scenes,methods]
    fig,ax=plt.subplots(figsize=(8.5,3.9))
    im=ax.imshow(values.to_numpy(),vmin=0,vmax=1,cmap="Blues",aspect="auto")
    ax.set_xticks(range(5),["Correlation","Linear\nfunctional","Interaction\nfunctional","Neural\nfunctional","Clock only"])
    ax.set_yticks(range(5),["Additive","Synergy","Redundant","Stress dependent","Unseen synergy"])
    for i in range(5):
        for j in range(5):
            v=values.iloc[i,j];ax.text(j,i,f"{v:.2f}",ha="center",va="center",color="white" if v>.65 else "#182a3a")
    ax.set_title("Synthetic upstream regulator recovery   precision at two",pad=12)
    fig.colorbar(im,ax=ax,pad=.03,label="Mean precision at two")
    fig.tight_layout();save(fig,"figure4_causal_recovery")


def regret():
    d=pd.read_csv(ROOT / "results/synthetic_summary.csv",keep_default_na=False)
    scenes=["additive","synergy","unseen_synergy"]
    models=["linear_function","interaction_function","neural_function","interaction_clock"]
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),gridspec_kw={"width_ratios":[1.7,1]})
    x=np.arange(3)
    for k,m in enumerate(models):
        sub=d[d.model==m].set_index("scenario").loc[scenes]
        axes[0].bar(x+(k-1.5)*.19,sub.policy_regret_mean,width=.19,color=COLORS[k],
                    label=["Linear function","Interaction function","Neural function","Clock only"][k])
    axes[0].set_xticks(x,["Additive","Synergy","Unseen synergy"])
    axes[0].set(ylabel="Functional policy regret",title="A  Unprotected planner  synthetic units")
    axes[0].legend(fontsize=7.8,ncol=2,loc="upper left")
    support=d[d.model=="interaction_function"].set_index("scenario").loc[scenes].pair01_supported_mean
    axes[1].bar(range(3),1-support,color=[COLORS[1],COLORS[1],COLORS[3]])
    axes[1].set_xticks(range(3),["Additive","Synergy","Unseen\nsynergy"])
    axes[1].set(ylim=(0,1.12),ylabel="Abstention fraction",title="B  Requested pair support gate")
    axes[1].text(2,1.02,"20 of 20",ha="center",fontsize=9)
    fig.tight_layout();save(fig,"figure5_support_and_regret")


if __name__=="__main__":
    architecture();public();synthetic_rmse();ranking();regret()
