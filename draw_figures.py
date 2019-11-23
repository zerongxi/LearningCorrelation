import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sequence_learners.temporal_mapping import TemporalMapping, TemporalMappingParameters
import data.utils


result_path = "./results_occurrence/"
figure_path = "./figures/"


def identify_best(path, prefix, k, window):
    fnames = [f for f in os.listdir(path) if f.startswith(prefix)]
    best = (0., "")

    for fname in fnames:
        fpath = os.path.join(path, fname)
        result = np.genfromtxt(fpath, delimiter=",")
        print(result[k, window], fname)
        if result[k, window] > best[0]:
            best = (result[k, window], fname[:-4])
    print(best)
    return best


def get_overview(k, window):
    datasets = [ele for ele in os.listdir(result_path) if not ele.startswith(".") and not ele.startswith("Web")]
    datasets.sort()
    approaches = ["informed", "local", "corr", "tmap", "deltalstm"]

    overview = np.zeros((len(datasets), len(approaches)), np.float32)

    for i in range(len(datasets)):
        for j in range(len(approaches)):
            overview[i, j] = identify_best(result_path+datasets[i], approaches[j], k, window)[0]

    buffer = "," + ",".join(datasets) + "\n"
    for i in range(len(approaches)):
        buffer += approaches[i] + "," + ",".join([str(ele) for ele in overview[:, i]]) + "\n"
    with open(figure_path+"overview_k_"+str(k)+"_window_"+str(window)+".csv", "w") as f:
        f.write(buffer)
    return


def draw_tmap_scatter():
    parameters =TemporalMappingParameters(
        kernel_mode="linear",
        kernel_window=32,
        kernel_diff=10.,
        alpha=5e-1,
        stop_criterion=5e-1 * 1e-2,
        n_epochs=2000,
        n_dims=2,
        norm="std",
        beta=None,
        min_repulse_distance=None,
        n_neighbors=None,
    )._asdict()
    parameters.update(dict(cuda_id=2))
    train, test = data.utils.load_seq("synthetic_nelements_5000")
    model = TemporalMapping(train, parameters)
    pts = model.index2vec[np.random.choice(model.index2vec.shape[0], 500, replace=False)]
    pts = model.index2vec
    limit = 2.
    index = np.logical_and(pts[:, 0] < limit, np.logical_and(pts[:, 0] > -limit, np.logical_and(
        pts[:, 1] < limit, pts[:, 1] > -limit
    )))
    pts = pts[index]

    plt.figure(figsize=(5, 5))
    plt.scatter(pts[:, 0], pts[:, 1], s=1)
    plt.xlabel("1st dimension")
    plt.ylabel("2nd dimension")
    plt.savefig(figure_path+"tamp_scatter.pdf")
    plt.show()
    return


def draw_illustration():
    pts = np.random.rand(300, 3) * 2 - 1
    pts = np.swapaxes(pts, 0, 1)

    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter(*pts, c="green")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    fig.savefig(figure_path+"illustration.pdf")
    fig.show()
    return






if __name__ == "__main__":
    # draw_illustration()
    get_overview(1, 128)
    # draw_tmap_scatter()
