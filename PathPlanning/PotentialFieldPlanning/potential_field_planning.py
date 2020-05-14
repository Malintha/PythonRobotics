"""

Potential Field based path planner

author: Atsushi Sakai (@Atsushi_twi)

Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf

"""

import numpy as np
import math
from numpy import linalg as la
import matplotlib.pyplot as plt

# Parameters
KP = 5.0  # attractive potential gain
ETA = 100.0  # repulsive potential gain
AREA_WIDTH = 30.0  # potential area width [m]

show_animation = True


def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, ox, oy, rr):
    # search nearest obstacle
    minid = -1
    dmin = float("inf")
    for i, _ in enumerate(ox):
        d = np.hypot(x - ox[i], y - oy[i])
        if dmin >= d:
            dmin = d
            minid = i

    # calc repulsive potential
    dq = np.hypot(x - ox[minid], y - oy[minid])

    if dq <= rr:
        if dq <= 0.1:
            dq = 0.1

        return 0.5 * ETA * (1.0 / dq - 1.0 / rr) ** 2
    else:
        return 0.0


def get_motion_model():
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion


#
# def potential_field_planning(sx, sy, gx, gy, ox, oy, reso, rr):
#
#     # calc potential field
#     pmap, minx, miny = calc_potential_field(gx, gy, ox, oy, reso, rr)
#
#     # search path
#     d = np.hypot(sx - gx, sy - gy)
#     ix = round((sx - minx) / reso)
#     iy = round((sy - miny) / reso)
#     gix = round((gx - minx) / reso)
#     giy = round((gy - miny) / reso)
#
#     if show_animation:
#         draw_heatmap(pmap)
#         # for stopping simulation with the esc key.
#         plt.gcf().canvas.mpl_connect('key_release_event',
#                 lambda event: [exit(0) if event.key == 'escape' else None])
#         plt.plot(ix, iy, "*k")
#         plt.plot(gix, giy, "*m")
#
#     rx, ry = [sx], [sy]
#     motion = get_motion_model()
#     while d >= reso:
#         minp = float("inf")
#         minix, miniy = -1, -1
#         for i, _ in enumerate(motion):
#             inx = int(ix + motion[i][0])
#             iny = int(iy + motion[i][1])
#             if inx >= len(pmap) or iny >= len(pmap[0]):
#                 p = float("inf")  # outside area
#             else:
#                 p = pmap[inx][iny]
#             if minp > p:
#                 minp = p
#                 minix = inx
#                 miniy = iny
#         ix = minix
#         iy = miniy
#         xp = ix * reso + minx
#         yp = iy * reso + miny
#         d = np.hypot(gx - xp, gy - yp)
#         rx.append(xp)
#         ry.append(yp)
#
#         if show_animation:
#             plt.plot(ix, iy, ".r")
#             plt.pause(0.005)
#
#     print("Goal!!")
#
#     return rx, ry


def calc_static_potential_field(ox, oy, gx, gy, res, rr):
    minx = min(ox) - AREA_WIDTH / 2.0
    miny = min(oy) - AREA_WIDTH / 2.0
    maxx = max(ox) + AREA_WIDTH / 2.0
    maxy = max(oy) + AREA_WIDTH / 2.0
    xw = int(round((maxx - minx) / res))
    yw = int(round((maxy - miny) / res))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    for ix in range(xw):
        x = ix * res + minx

        for iy in range(yw):
            y = iy * res + miny
            ug = calc_attractive_potential(x, y, gx, gy)
            uo = calc_repulsive_potential(x, y, ox, oy, rr)
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap, minx, miny, maxx, maxy


def get_interaction_potential(posA, posB):
    a = 0.2
    b = 2
    ca = 8
    cr = 1
    dis = la.norm(np.array(posA) - np.array(posB))
    att_pot = -a * math.exp(-dis / ca)
    rep_pot = b * math.exp(-dis / cr)
    return att_pot + rep_pot


def cal_dynamic_potential_field(curr_pos, id, xw, yw):
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]
    for i in range(yw):
        for j in range(xw):
            for r in range(len(curr_pos)):
                if r == id:
                    continue
                pmap[j][i] += 3*get_interaction_potential([i, j], curr_pos[r]) + 2
    return pmap


def calc_next_position(id, curr_pos, gx, gy, ox, oy, res, rr):
    pmap, minx, miny, maxx, maxy = calc_static_potential_field(ox, oy, gx, gy, res, rr)
    gix = round((gx - minx) / res)
    giy = round((gy - miny) / res)
    xw = int(round((maxx - minx) / res))
    yw = int(round((maxy - miny) / res))

    pdmap = cal_dynamic_potential_field(curr_pos, id, xw, yw)
    synthesized_map = np.array(pmap) + np.array(pdmap)
    if show_animation:
        draw_heatmap(pmap)
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(gix, giy, "*m")

        ix = round((curr_pos[id][0] - minx) / res)
        iy = round((curr_pos[id][1] - miny) / res)
        plt.plot(ix, iy, "*k")

    motion = get_motion_model()
    minp = float("inf")
    minix, miniy = -1, -1
    for i, _ in enumerate(motion):
        inx = int(ix + motion[i][0])
        iny = int(iy + motion[i][1])
        if inx >= len(pmap) or iny >= len(pmap[0]):
            p = float("inf")  # outside area
        else:
            p = synthesized_map[inx][iny]
        if minp > p:
            minp = p
            minix = inx
            miniy = iny
    ix = minix
    iy = miniy
    xp = ix * res + minx
    yp = iy * res + miny
    curr_pos[id] = [xp, yp]
    # d = np.hypot(gx - xp, gy - yp)
    # rx.append(xp)
    # ry.append(yp)

    if show_animation:
    #     plt.plot(xp, yp, ".r")
        plt.pause(0.001)
    return curr_pos


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


def main():
    print("potential_field_planning start")

    curr_pos = [[0, 10],
                [0, 5],
                [5, 10]]

    gx = 30.0  # goal x position [m]
    gy = 30.0  # goal y position [m]
    grid_size = 0.5  # potential grid size [m]
    robot_radius = 2.0  # robot radius [m]

    ox = [15.0, 5.0, 20.0, 25.0]  # obstacle x position list [m]
    oy = [25.0, 15.0, 26.0, 25.0]  # obstacle y position list [m]

    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    # path generation
    for it in range(20):
        for r in range(len(curr_pos)):
            curr_pos = calc_next_position(r, curr_pos, gx, gy, ox, oy, grid_size, robot_radius)

    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
