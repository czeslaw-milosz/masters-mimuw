from sigProfilerPlotting import sigplot


figs = sigplot.plotSBS(
    "./data/cosmic/cosmic.csv", 
    "tmp/", "project", plot_type="96", savefig_format="png", dpi=200, percentage=True, return_figs=True
)
for signature_name, fig in figs.items():
    fig.savefig(f"./data/cosmic/plots/{signature_name}.png", format="png", dpi=200)

# sigPlt.plotSBS("./data/cosmic/cosmic.csv", 
#                "./data/cosmic/signature_plots/", "", plot_type="96", savefig_format="png", dpi=200, percentage=True
#                )
