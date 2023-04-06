import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

# set plot params
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams["font.family"] = 'serif'
plt.rcParams['font.size'] = 16

def sparsity():
    yield .80
    yield .75
    yield .45
    yield .35

def create_jacobian_figures(model_size=25, sg=sparsity(), surf_percentage=0.85):
    random.seed(time.time())
    gas_len = int(surf_percentage * model_size)
    surf_len = model_size - gas_len
    colors = ["#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"]
    markers = ["s", "P", "^", "X"]
    gas_gas_pts = [(i, j) for i in range(gas_len) for j in range(gas_len) if i != j]
    surf_surf_pts = [(i, j) for i in range(gas_len, model_size) for j in range(gas_len,model_size) if i != j]
    gas_surf_pts = [(i, j) for i in range(gas_len) for j in range(gas_len,model_size) if i != j]
    surf_gas_pts = [(i, j) for i in range(gas_len, model_size) for j in range(gas_len) if i != j]
    diag = [(i, i) for i in range(model_size)]
    pts_list = [gas_gas_pts, surf_surf_pts, surf_gas_pts, gas_surf_pts]
    lens = [len(pts) for pts in pts_list]
    # flip y chords:
    for pts in pts_list:
        for k in range(len(pts)):
            i, j = pts[k]
            pts[k] = (i, model_size-j)
    for k in range(len(diag)):
        i, j = diag[k]
        diag[k] = (i, model_size-j)
    # create sparsity reduction closure
    def reduce_sparsity(sp):
        npts = []
        for i in range(len(pts_list)):
            pts = pts_list[i]
            pts_olen = lens[i]
            pts_newlen = int(sp * pts_olen)
            random.shuffle(pts)
            npts.append(pts[:pts_newlen])
        return npts
    # create gas phase figure
    # create figure
    fig, ax = plt.subplots(1, 1)
    # plot diagonal
    x, y = zip(*gas_gas_pts)
    ax.plot(x, y, color=colors[0], linestyle="", marker=markers[0])
    # plot diagonal
    x, y = zip(*diag[:gas_len])
    ax.plot(x, y, color="#8856a7", linestyle="", marker="o")
    # ax.axis("off")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tick_params(left=False, bottom=False)
    fig.tight_layout()
    plt.savefig(f"usncm_figures/gas-phase.jpg")
    plt.close()
    # closure for creating plot
    def create_plot(name, pts_list):
        # create figure
        fig, ax = plt.subplots(1, 1)
        for i, pts in enumerate(pts_list):
            x, y = zip(*pts)
            ax.plot(x, y, color=colors[i], linestyle="", marker=markers[i])
        # plot diagonal
        x, y = zip(*diag)
        ax.plot(x, y, color="#8856a7", linestyle="", marker="o")
        # ax.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.tick_params(left=False, bottom=False)
        fig.tight_layout()
        plt.savefig(f"usncm_figures/{name}.jpg")
        plt.close()

    # create dense plot
    create_plot("surf-phase", pts_list)

    # create next level
    sp = next(sg)
    pts_list = reduce_sparsity(sp)
    create_plot("moles", pts_list)

    # create next level
    sp = next(sg)
    pts_list = reduce_sparsity(sp)
    create_plot("nfo-ntb", pts_list)

    # create next level
    sp = next(sg)
    pts_list = reduce_sparsity(sp)
    create_plot("nfo-ntb", pts_list)

    # create next level
    sp = next(sg)
    pts_list = reduce_sparsity(sp)
    create_plot("threshold", pts_list)

    #create next reactor in network
    copy_pts = copy.deepcopy(pts_list)
    for q, pts in enumerate(copy_pts):
        for i in range(len(pts)):
            j, k = pts[i]
            pts[i] = (j + model_size, k - model_size)
        pts_list[q] += copy_pts[q]
    # update diag
    for i in range(len(diag)):
        j, k = diag[i]
        diag.append((j + model_size, k - model_size))
    # create cross reactor data
    cross_data = []
    for i in range(model_size, 2 * model_size):
        for j in range(model_size):
            cross_data.append((i, j))
    for i in range(model_size):
        for j in range(-1, -model_size, -1):
            cross_data.append((i, j))
    random.shuffle(cross_data)
    cross_data = cross_data[:int(len(cross_data) * sp)]
    # create figure
    fig, ax = plt.subplots(1, 1)
    for i, pts in enumerate(pts_list):
        x, y = zip(*pts)
        ax.plot(x, y, color=colors[i], linestyle="", marker=markers[i])
    # plot diagonal
    x, y = zip(*cross_data)
    ax.plot(x, y, color="#addd8e", linestyle="", marker="v")
    x, y = zip(*diag)
    ax.plot(x, y, color="#8856a7", linestyle="", marker="o")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tick_params(left=False, bottom=False)
    fig.tight_layout()
    plt.savefig(f"usncm_figures/network.jpg")
    plt.close()



create_jacobian_figures()
