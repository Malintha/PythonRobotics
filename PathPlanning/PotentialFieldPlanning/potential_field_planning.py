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


def calc_goal_potential(pos, goal):
    return 3 * math.exp(la.norm(np.array(pos) - np.array(goal)) / 30)


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
            ug = calc_goal_potential([x, y], [gx, gy])
            uo = calc_repulsive_potential(x, y, ox, oy, rr)
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap, minx, miny, maxx, maxy


def get_interaction_potential(posA, posB):
    a = 1
    b = 2
    ca = 10
    cr = 3
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


def calc_next_position(id, curr_pos, gx, gy, ox, oy, res, pmap, xvals, yvals):
    minx = xvals[0]
    maxx = xvals[1]
    miny = yvals[0]
    maxy = yvals[1]
    xw = int(round((maxx - minx) / res))
    yw = int(round((maxy - miny) / res))

    pdmap = cal_dynamic_potential_field(curr_pos, id, xw, yw)
    synthesized_map = np.array(pmap) + np.array(pdmap)

    ix = round((curr_pos[id][0] - minx) / res)
    iy = round((curr_pos[id][1] - miny) / res)
    if show_animation:
        plt.plot(ix, iy, "*r")

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

    if show_animation:
        plt.pause(0.001)
    return curr_pos


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


def main():
    print("potential_field_planning start")

    curr_pos = [[5, 8],
                [3, 5],
                [8, 6]]

    gx = 30.0  # goal x position [m]
    gy = 30.0  # goal y position [m]

    grid_size = 0.5  # potential grid size [m]
    robot_radius = 5.0  # robot radius [m]

    ox = [15.0, 5.0, 20.0, 25.0]  # obstacle x position list [m]
    oy = [25.0, 15.0, 26.0, 25.0]  # obstacle y position list [m]

    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    pmap, minx, miny, maxx, maxy = calc_static_potential_field(ox, oy, gx, gy, grid_size, robot_radius)

    gix = round((gx - minx) / grid_size)
    giy = round((gy - miny) / grid_size)

    if show_animation:
        draw_heatmap(pmap)
        plt.plot(gix, giy, "*m")
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

    xvals = [minx, maxx]
    yvals = [miny, maxy]

    # path generation
    for it in range(60):
        for r in range(len(curr_pos)):
            curr_pos = calc_next_position(r, curr_pos, gx, gy, ox, oy, grid_size, pmap, xvals, yvals)

    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
