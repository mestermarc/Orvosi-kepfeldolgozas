import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec

from skimage import filters

def data_regularize(data, type="spherical", divs=10):
    limits = np.array([
        [min(data[:, 0]), max(data[:, 0])],
        [min(data[:, 1]), max(data[:, 1])],
        [min(data[:, 2]), max(data[:, 2])]])

    regularized = []

    if type == "cubic":  # take mean from points in the cube

        X = np.linspace(*limits[0], num=divs)
        Y = np.linspace(*limits[1], num=divs)
        Z = np.linspace(*limits[2], num=divs)

        for i in range(divs - 1):
            for j in range(divs - 1):
                for k in range(divs - 1):
                    points_in_sector = []
                    for point in data:
                        if (point[0] >= X[i] and point[0] < X[i + 1] and
                                point[1] >= Y[j] and point[1] < Y[j + 1] and
                                point[2] >= Z[k] and point[2] < Z[k + 1]):
                            points_in_sector.append(point)
                    if len(points_in_sector) > 0:
                        regularized.append(np.mean(np.array(points_in_sector), axis=0))

    elif type == "spherical":  # take mean from points in the sector
        divs_u = divs
        divs_v = divs * 2

        center = np.array([
            0.5 * (limits[0, 0] + limits[0, 1]),
            0.5 * (limits[1, 0] + limits[1, 1]),
            0.5 * (limits[2, 0] + limits[2, 1])])
        d_c = data - center

        # spherical coordinates around center
        r_s = np.sqrt(d_c[:, 0] ** 2. + d_c[:, 1] ** 2. + d_c[:, 2] ** 2.)
        d_s = np.array([
            r_s,
            np.arccos(d_c[:, 2] / r_s),
            np.arctan2(d_c[:, 1], d_c[:, 0])]).T

        u = np.linspace(0, np.pi, num=divs_u)
        v = np.linspace(-np.pi, np.pi, num=divs_v)

        for i in range(divs_u - 1):
            for j in range(divs_v - 1):
                points_in_sector = []
                for k, point in enumerate(d_s):
                    if (point[1] >= u[i] and point[1] < u[i + 1] and
                            point[2] >= v[j] and point[2] < v[j + 1]):
                        points_in_sector.append(data[k])

                if len(points_in_sector) > 0:
                    regularized.append(np.mean(np.array(points_in_sector), axis=0))
    # Other strategy of finding mean values in sectors
    #                    p_sec = np.array(points_in_sector)
    #                    R = np.mean(p_sec[:,0])
    #                    U = (u[i] + u[i+1])*0.5
    #                    V = (v[j] + v[j+1])*0.5
    #                    x = R*math.sin(U)*math.cos(V)
    #                    y = R*math.sin(U)*math.sin(V)
    #                    z = R*math.cos(U)
    #                    regularized.append(center + np.array([x,y,z]))
    return np.array(regularized)


# https://github.com/minillinim/ellipsoid
def ellipsoid_plot(center, radii, rotation, ax, plot_axes=False, cage_color='b', cage_alpha=0.2):
    """Plot an ellipsoid"""

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    if plot_axes:
        # make some purdy axes
        axes = np.array([[radii[0], 0.0, 0.0],
                         [0.0, radii[1], 0.0],
                         [0.0, 0.0, radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)

        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cage_color)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color=cage_color, alpha=cage_alpha)


# http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
# for arbitrary axes
def ellipsoid_fit(X):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                  x * x + z * z - 2 * y * y,
                  2 * x * y,
                  2 * x * z,
                  2 * y * z,
                  2 * x,
                  2 * y,
                  2 * z,
                  1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T  # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    chi2 = (1 - (D.T.dot(u)) / d2)**2

    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)
    print("chi2 avg:", np.average(chi2))
    acc = get_fitting_accuracy(radii[0], radii[1], radii[2], chi2)

    ecc = calc_eccentricity(radii[0], radii[1], radii[2])

    return center, evecs, radii, v, acc, ecc


import numpy as np



def get_fitting_accuracy(x_axis, y_axis, z_axis, llsq):
    avg_axis_size = (x_axis+y_axis+z_axis)/3
    avg_llsq = np.average(llsq)
    rate = avg_axis_size/avg_llsq
    accuracy = round(rate/4090, 2)*100
    print("accuracy:{}%".format(accuracy))
    return accuracy

def findEccentricity(A, B):
    if(A==B):
        return 0
    semiMajor = A * A
    semiMinor = B * B

    ans = math.sqrt(1 - semiMinor / semiMajor)

    return ans

def calc_eccentricity(x_axis, y_axis, z_axis):
    semi_major_axis = max(x_axis, y_axis, z_axis)
    eccentricities = []
    eccentricities.append(findEccentricity(semi_major_axis,x_axis))
    eccentricities.append(findEccentricity(semi_major_axis,y_axis))
    eccentricities.append(findEccentricity(semi_major_axis,z_axis))

    eccentricities2 = [ecc for ecc in eccentricities if ecc >0]
    print("ecc length: ", len(eccentricities2))
    return eccentricities2



def ellipsoid_acc(tumor):
    edges = []

    for slices in tumor:
        edge = filters.roberts(slices)
        edge[edge > 0] = 1
        int_array = edge.astype(int)
        edges.append(int_array)

    pontfelho = []
    slicecounter = 1
    for edge in edges:
        middlex = int(round(len(edge) / 2, 0))
        kieg = 0
        if middlex < len(edge) / 2:
            kieg = 1
        for row in range(0, len(edge)):
            middley = int(round(len(edge[slicecounter]) / 2, 0))
            for column in range(0, len(edge[slicecounter])):
                # print(row, column, slicecounter)
                if edge[row][column] > 0:
                    pontfelho.append((row - middlex, column - middley, slicecounter * 2.31))
        slicecounter += 1

    print("pontfelho hossza:", len(pontfelho))

    xp = []
    yp = []
    zp = []

    for pont in pontfelho:
        xp.append(pont[0])
        yp.append(pont[1])
        zp.append(pont[2])

    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(xp, yp, zp, zdir='z', s=20, c='b', rasterized=True)
    ax.set_xlabel('x')

    import numpy as np
    import plotly.offline as go_offline
    import plotly.graph_objects as go

    # CREATING 3D TERRAIN MODEL

    # fig=go.Figure()
    # fig.add_trace(go.Surface(z=zp,x=xp,y=yp))
    # fig.update_layout(scene=dict(aspectratio=dict(x=2, y=2, z=0.5),xaxis = dict(range=[-3,3],),yaxis = dict(range=[-3,3])))

    # 3d fit
    data = np.array(pontfelho)

    data2 = data_regularize(data, divs=8)

    center, evecs, radii, v = ellipsoid_fit(data2)

    data_centered = data - center.T
    data_centered_regularized = data2 - center.T

    a, b, c = radii
    r = (a * b * c) ** (1. / 3.)
    D = np.array([[r / a, 0., 0.], [0., r / b, 0.], [0., 0., r / c]])
    # http://www.cs.brandeis.edu/~cs155/Lecture_07_6.pdf
    # affine transformation from ellipsoid to sphere (translation excluded)
    TR = evecs.dot(D).dot(evecs.T)
    data_on_sphere = TR.dot(data_centered_regularized.T).T

    ax2 = fig.add_subplot(122, projection='3d')

    # hack  for equal axes
    # ax.set_aspect('equal')
    # for direction in (-1, 1):
    #     for point in np.diag(direction * np.max(data) * np.array([1, 1, 1])):
    #         ax.plot([point[0]], [point[1]], [point[2]], 'w')

    ax2.scatter(data_centered[:, 0], data_centered[:, 1], data_centered[:, 2], marker='o', color='b')
    # ax.scatter(data_centered_regularized[:, 0], data_centered_regularized[:, 1],
    #            data_centered_regularized[:, 2], marker='o', color='b')
    ellipsoid_plot([0, 0, 0], radii, evecs, ax=ax2, plot_axes=True, cage_color='g')
    # ellipsoid_plot([0, 0, 0], [r, r, r], evecs, ax=ax, plot_axes=True, cage_color='orange')

    plt.show()




from skimage import morphology

def ellipsoid_plotting(tumorsl, tumor):
    edges = []

    for slices in tumorsl:
        slices = morphology.remove_small_objects(slices.astype(bool), min_size=3)

        edge = filters.roberts(slices)
        edge[edge > 0] = 1
        int_array = edge.astype(int)
        edges.append(int_array)

    pontfelho = []
    slicecounter = 1
    for edge in edges:
        middlex = int(round(len(edge) / 2, 0))
        kieg = 0
        if middlex < len(edge) / 2:
            kieg = 1
        for row in range(0, len(edge)):
            middley = int(round(len(edge[slicecounter]) / 2, 0))
            for column in range(0, len(edge[slicecounter])):
                # print(row, column, slicecounter)
                if edge[row][column] > 0:
                    pontfelho.append((row - middlex, column - middley, slicecounter * 2.31))
        slicecounter += 1

    print("pontfelho hossza:", len(pontfelho))

    xp = []
    yp = []
    zp = []

    for pont in pontfelho:
        xp.append(pont[0])
        yp.append(pont[1])
        zp.append(pont[2])

    fig = plt.figure(figsize=(10, 10), dpi=80)
    gs = GridSpec(2, 4)  # 2 rows, 4 columns


    ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
    ax1.imshow(tumorsl[0], cmap="bone",interpolation="nearest")  # crop_img = img[y:y+h, x:x+w
    #ax1.title.set_text("#{}".format(num))
    ax1.set_axis_off()

    ax2 = fig.add_subplot(gs[0, 1])  # First row, second column
    ax2.imshow(tumorsl[1], cmap="bone", interpolation="nearest")  # crop_img = img[y:y+h, x:x+w
    # ax1.title.set_text("#{}".format(num))
    ax2.set_axis_off()
    ax3 = fig.add_subplot(gs[0, 2])  # First row, third column
    ax3.imshow(tumorsl[2], cmap="bone", interpolation="nearest")  # crop_img = img[y:y+h, x:x+w
    # ax1.title.set_text("#{}".format(num))
    ax3.set_axis_off()
    ax4 = fig.add_subplot(gs[0, 3])  # First row, third column
    ax4.imshow(tumorsl[3], cmap="bone", interpolation="nearest")  # crop_img = img[y:y+h, x:x+w
    # ax1.title.set_text("#{}".format(num))
    ax4.set_axis_off()

    ax = fig.add_subplot(gs[1, :-2], projection='3d')
    ax.scatter(xp, yp, zp, zdir='z', s=20, c='b', rasterized=True)
    ax.set_xlabel('x')

    import numpy as np
    import plotly.offline as go_offline
    import plotly.graph_objects as go

    # CREATING 3D TERRAIN MODEL

    # fig=go.Figure()
    # fig.add_trace(go.Surface(z=zp,x=xp,y=yp))
    # fig.update_layout(scene=dict(aspectratio=dict(x=2, y=2, z=0.5),xaxis = dict(range=[-3,3],),yaxis = dict(range=[-3,3])))

    # 3d fit
    data = np.array(pontfelho)

    data2 = data_regularize(data, divs=8)

    center, evecs, radii, v, acc, ecc = ellipsoid_fit(data2)

    data_centered = data - center.T
    data_centered_regularized = data2 - center.T

    a, b, c = radii
    r = (a * b * c) ** (1. / 3.)
    D = np.array([[r / a, 0., 0.], [0., r / b, 0.], [0., 0., r / c]])
    # http://www.cs.brandeis.edu/~cs155/Lecture_07_6.pdf
    # affine transformation from ellipsoid to sphere (translation excluded)
    TR = evecs.dot(D).dot(evecs.T)
    data_on_sphere = TR.dot(data_centered_regularized.T).T

    ax2 = fig.add_subplot(gs[1, 2:], projection='3d')

    # hack  for equal axes
    # ax.set_aspect('equal')
    # for direction in (-1, 1):
    #     for point in np.diag(direction * np.max(data) * np.array([1, 1, 1])):
    #         ax.plot([point[0]], [point[1]], [point[2]], 'w')

    ax2.scatter(data_centered[:, 0], data_centered[:, 1], data_centered[:, 2], marker='o', color='b')
    # ax.scatter(data_centered_regularized[:, 0], data_centered_regularized[:, 1],
    #            data_centered_regularized[:, 2], marker='o', color='b')
    ellipsoid_plot([0, 0, 0], radii, evecs, ax=ax2, plot_axes=True, cage_color='g')
    # ellipsoid_plot([0, 0, 0], [r, r, r], evecs, ax=ax, plot_axes=True, cage_color='orange')



    title = "Suspicious form: ID:{}, lenght:{}, accuracy={}, ellips valid:{}%\neccentricities are: {}, {}"\
                .format(tumor.getId(), tumor.getLenght(), tumor.get_proba(), acc, ecc[0],ecc[1]) + "\n"
    fig.suptitle(title, fontsize=16)

    plt.show()
