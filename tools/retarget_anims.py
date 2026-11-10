#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retarget_anims.py - fit CS 1.6 player animation SMDs to this model's skeleton.

Why: GoldSrc animations store bone *positions* (= bone lengths) as well as
rotations.  The stock CS 1.6 player animations were made for the stock
skeleton (longer arms, wider shoulders, longer neck, different legs).  If they
are compiled as-is with a model whose reference skeleton has other proportions,
every body part is dragged to the stock joint positions and the mesh gets
stretched / torn at every joint.

What it does, for every animation SMD:
  * keeps only the bones that exist in the model's reference skeleton,
  * keeps every bone ROTATION from the animation,
  * replaces every bone POSITION by the one from the model's reference pose,
  * scales the movement of the pelvis by the leg-length ratio so the feet stay
    on the ground (and do not slide) in standing and crouching animations,
  * bones that the animation does not have (e.g. "L Calf_Sub") keep their
    reference pose.

Usage:
    python tools/retarget_anims.py <reference.smd> <src_anims_dir> <dst_anims_dir> [--leg-scale K]

<reference.smd> should be the reference mesh that contains the feet (0.smd here);
the leg scale is measured from it automatically unless --leg-scale is given.

Pure Python 3, no extra modules needed.
"""
import math, os, sys

HULL_STAND = 36.0   # player origin above the floor while standing
HULL_DUCK = 18.0    # ... and while ducking


def parse(path):
    nodes, frames, rest = [], [], []
    with open(path, encoding='latin-1') as f:
        lines = f.read().replace('\r', '').split('\n')
    mode, cur, i = None, None, 0
    while i < len(lines):
        s = lines[i].strip()
        i += 1
        if mode is None:
            if s == 'nodes':
                mode = 'nodes'
            elif s == 'skeleton':
                mode = 'skel'
            elif s == 'triangles':
                break
        elif mode == 'nodes':
            if s == 'end':
                mode = None
                continue
            a = s.find('"'); b = s.rfind('"')
            nodes.append((int(s[:a]), s[a + 1:b], int(s[b + 1:])))
        elif mode == 'skel':
            if s == 'end':
                mode = None
            elif s.startswith('time'):
                cur = {}
                frames.append(cur)
            elif s:
                p = s.split()
                cur[int(p[0])] = [float(x) for x in p[1:7]]
    return nodes, frames


def rot(rx, ry, rz):
    """SMD euler -> 3x3 matrix, R = Rz * Ry * Rx."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return [[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx]]


def mv(m, v):
    return [m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2] for r in range(3)]


def mm(a, b):
    return [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)] for r in range(3)]


def world(nodes, frame):
    """world (R, T) of every bone of one frame."""
    R, T = {}, {}
    for idx, _, par in sorted(nodes):
        v = frame[idx]
        r = rot(*v[3:6])
        if par < 0:
            R[idx], T[idx] = r, v[:3]
        else:
            R[idx] = mm(R[par], r)
            t = mv(R[par], v[:3])
            T[idx] = [t[k] + T[par][k] for k in range(3)]
    return R, T


def leg_length(nodes, frame, side='L'):
    """thigh + calf + ankle height (hip joint down to the toe bone) of a skeleton."""
    by = {n: i for i, n, _ in nodes}
    try:
        calf = frame[by['Bip01 %s Calf' % side]][:3]
        foot = frame[by['Bip01 %s Foot' % side]][:3]
        toe = frame[by['Bip01 %s Toe0' % side]][:3]
    except KeyError:
        return None
    n = lambda v: math.sqrt(sum(x * x for x in v))
    return n(calf) + n(foot) + abs(toe[0])


def sole_thickness(path, nodes, frame, side='L'):
    """how far the mesh reaches below the toe bone in the reference pose."""
    by = {n: i for i, n, _ in nodes}
    if 'Bip01 %s Toe0' % side not in by:
        return 0.0
    _, T = world(nodes, frame)
    toe_z = T[by['Bip01 %s Toe0' % side]][2]
    zmin = None
    with open(path, encoding='latin-1') as f:
        tri = False
        for line in f:
            if not tri:
                tri = line.strip() == 'triangles'
                continue
            p = line.split()
            if len(p) >= 9:
                try:
                    z = float(p[3])
                except ValueError:
                    continue
                zmin = z if zmin is None else min(zmin, z)
    return max(0.0, toe_z - zmin) if zmin is not None else 0.0


STOCK_SOLE = 0.1    # the stock CS skeleton has its toe bones ~0.1 above the floor


def retarget(ref_nodes, ref_pose, src_path, dst_path, k):
    nodes, frames = parse(src_path)
    if not frames:
        return False
    by = {n: i for i, n, _ in nodes}
    root = [i for i, n, p in ref_nodes if p < 0][0]
    root_name = [n for i, n, p in ref_nodes if i == root][0]
    pelvis = [i for i, n, p in ref_nodes if n.endswith('Pelvis') and p == root]
    pelvis = pelvis[0] if pelvis else None
    name = os.path.basename(src_path).lower()
    hull = HULL_DUCK if name.startswith('crouch') else HULL_STAND
    out = ['version 1', 'nodes']
    for i, n, p in ref_nodes:
        out.append('%3d "%s" %d' % (i, n, p))
    out += ['end', 'skeleton']
    prev = {}
    for t, fr in enumerate(frames):
        out.append('time %d' % t)
        for i, n, p in ref_nodes:
            v = list(ref_pose[i])
            if n in by and by[n] in fr:
                a = fr[by[n]]
                v[3:6] = a[3:6]
                if i == root:
                    v[0:3] = a[0:3]
            if i == root and root_name in by:
                # where is the pelvis in the original animation?
                src_root = fr[by[root_name]]
                Rr = rot(*src_root[3:6])
                pn = ref_nodes[pelvis][1] if pelvis is not None else None
                if pn in by and by[pn] in fr:
                    off = mv(Rr, fr[by[pn]][:3])
                else:
                    off = [0.0, 0.0, 0.0]
                P = [src_root[j] + off[j] for j in range(3)]
                # scale it around the point on the floor below the player origin
                P = [P[0] * k, P[1] * k, (P[2] + hull) * k - hull]
                moff = mv(Rr, ref_pose[pelvis][:3]) if pelvis is not None else [0.0, 0.0, 0.0]
                v[0:3] = [P[j] - moff[j] for j in range(3)]
            # keep euler angles continuous between frames (avoids 2*pi jumps)
            if i in prev:
                for j in range(3, 6):
                    while v[j] - prev[i][j] > math.pi:
                        v[j] -= 2 * math.pi
                    while v[j] - prev[i][j] < -math.pi:
                        v[j] += 2 * math.pi
            prev[i] = v
            out.append('%3d   %.6f %.6f %.6f %.6f %.6f %.6f' % tuple([i] + v))
    out.append('end')
    with open(dst_path, 'w', newline='\n') as f:
        f.write('\n'.join(out) + '\n')
    return True


def main(argv):
    argv = list(argv)
    k = None
    if '--leg-scale' in argv:
        j = argv.index('--leg-scale')
        k = float(argv[j + 1])
        del argv[j:j + 2]
    if len(argv) != 3:
        print(__doc__)
        return 1
    ref_path, src, dst = argv
    ref_nodes, ref_frames = parse(ref_path)
    ref_pose = ref_frames[0]
    files = sorted(f for f in os.listdir(src) if f.lower().endswith('.smd'))
    if k is None:
        # leg-length ratio: model / skeleton the animations were made for
        ml = leg_length(ref_nodes, ref_pose)
        k = 1.0
        for f in files:
            n, fr = parse(os.path.join(src, f))
            al = leg_length(n, fr[0]) if fr else None
            if ml and al:
                ml += sole_thickness(ref_path, ref_nodes, ref_pose)
                k = ml / (al + STOCK_SOLE)
                break
    print('leg scale: %.4f' % k)
    os.makedirs(dst, exist_ok=True)
    done = 0
    for f in files:
        if retarget(ref_nodes, ref_pose, os.path.join(src, f), os.path.join(dst, f), k):
            done += 1
    print('retargeted %d animation(s) -> %s' % (done, dst))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
