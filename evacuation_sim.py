import numpy as np
from vedo import Plotter, Plane, Sphere, Line

np.random.seed(7)

# -----------------------------
# setup
# -----------------------------
AGENT_COUNT = 24
LEVEL_GAP = 4.0
LEVELS = [0.0, LEVEL_GAP, 2.0 * LEVEL_GAP]

TIME_STEP = 0.06
MAX_FRAMES = 5000

MOVE_GAIN = 0.9
AVOID_GAIN = 0.55
HEIGHT_GAIN = 0.28

PERSONAL_SPACE = 1.1
RAMP_RADIUS = 1.4

DOORS = np.array([
    [7.0, -4.5, 0.0],
    [-7.0, -4.5, 0.0]
], dtype=float)

# ramps go downward from upper floor to lower floor
UPPER_RAMP_START = np.array([-3.0, 3.0], dtype=float)
UPPER_RAMP_END   = np.array([-6.0, -2.0], dtype=float)

LOWER_RAMP_START = np.array([3.0, 3.0], dtype=float)
LOWER_RAMP_END   = np.array([6.0, -2.0], dtype=float)

# -----------------------------
# geometry helpers
# -----------------------------
def clamp01(val):
    return max(0.0, min(1.0, val))


def dist_to_line_segment(pt, a, b):
    seg = b - a
    denom = np.dot(seg, seg)
    if denom == 0:
        return np.linalg.norm(pt - a)

    u = clamp01(np.dot(pt - a, seg) / denom)
    nearest = a + u * seg
    return np.linalg.norm(pt - nearest)


def lerp_height(pt_xy, a, b, z_start, z_end):
    seg = b - a
    denom = np.dot(seg, seg)
    if denom == 0:
        return z_start

    u = clamp01(np.dot(pt_xy - a, seg) / denom)
    return z_start + u * (z_end - z_start)


def on_ramp(pt_xy, ramp_a, ramp_b):
    return dist_to_line_segment(pt_xy, ramp_a, ramp_b) <= RAMP_RADIUS


def floor_band(z):
    if z > 1.5 * LEVEL_GAP:
        return 2
    if z > 0.5 * LEVEL_GAP:
        return 1
    return 0


# -----------------------------
# surface projection
# -----------------------------
def target_surface_height(x, y, previous_z):
    q = np.array([x, y], dtype=float)

    # top floor: can only use upper ramp down to middle
    if previous_z > 1.5 * LEVEL_GAP:
        if on_ramp(q, UPPER_RAMP_START, UPPER_RAMP_END):
            return lerp_height(q, UPPER_RAMP_START, UPPER_RAMP_END, 2.0 * LEVEL_GAP, LEVEL_GAP)
        return 2.0 * LEVEL_GAP

    # middle floor: can use lower ramp to ground
    if previous_z > 0.5 * LEVEL_GAP:
        if on_ramp(q, LOWER_RAMP_START, LOWER_RAMP_END):
            return lerp_height(q, LOWER_RAMP_START, LOWER_RAMP_END, LEVEL_GAP, 0.0)
        return LEVEL_GAP

    # ground floor
    return 0.0


# -----------------------------
# navigation logic
# -----------------------------
def pick_waypoint(agent):
    x, y, z = agent
    xy = agent[:2]

    if z > 1.5 * LEVEL_GAP:
        if on_ramp(xy, UPPER_RAMP_START, UPPER_RAMP_END):
            return np.array([UPPER_RAMP_END[0], UPPER_RAMP_END[1], LEVEL_GAP], dtype=float)
        return np.array([UPPER_RAMP_START[0], UPPER_RAMP_START[1], 2.0 * LEVEL_GAP], dtype=float)

    if z > 0.5 * LEVEL_GAP:
        if on_ramp(xy, LOWER_RAMP_START, LOWER_RAMP_END):
            return np.array([LOWER_RAMP_END[0], LOWER_RAMP_END[1], 0.0], dtype=float)
        return np.array([LOWER_RAMP_START[0], LOWER_RAMP_START[1], LEVEL_GAP], dtype=float)

    door_dists = np.linalg.norm(DOORS - agent, axis=1)
    return DOORS[np.argmin(door_dists)]


def desired_motion(agent):
    target = pick_waypoint(agent)
    vec = target - agent
    vec[2] = 0.0

    mag = np.linalg.norm(vec)
    if mag < 1e-8:
        return np.zeros(3)

    return vec / mag


# -----------------------------
# crowd interaction
# -----------------------------
def separation_push(idx, crowd):
    me = crowd[idx]
    push = np.zeros(3)

    for j, other in enumerate(crowd):
        if j == idx:
            continue

        diff = me - other
        diff[2] = 0.0
        d = np.linalg.norm(diff)

        if 1e-6 < d < PERSONAL_SPACE:
            push += diff / (d * d)

    return push


def queue_slowdown(idx, crowd):
    me = crowd[idx]
    me_xy = me[:2]
    factor = 1.0

    for j, other in enumerate(crowd):
        if j == idx:
            continue

        same_upper = on_ramp(me_xy, UPPER_RAMP_START, UPPER_RAMP_END) and on_ramp(other[:2], UPPER_RAMP_START, UPPER_RAMP_END)
        same_lower = on_ramp(me_xy, LOWER_RAMP_START, LOWER_RAMP_END) and on_ramp(other[:2], LOWER_RAMP_START, LOWER_RAMP_END)

        if same_upper or same_lower:
            if np.linalg.norm(me - other) < 1.6:
                factor = 0.35

    return factor


# -----------------------------
# update rule
# -----------------------------
def advance_agents(crowd):
    updated = crowd.copy()

    for i, person in enumerate(crowd):
        nav = desired_motion(person)
        sep = separation_push(i, crowd)
        slow = queue_slowdown(i, crowd)

        velocity = slow * (MOVE_GAIN * nav + AVOID_GAIN * sep)
        proposal = person + TIME_STEP * velocity

        # snap gently toward valid floor/ramp height
        surf_z = target_surface_height(proposal[0], proposal[1], person[2])
        proposal[2] = person[2] + HEIGHT_GAIN * (surf_z - person[2])

        updated[i] = proposal

    return updated


# -----------------------------
# spawn agents
# -----------------------------
def spawn_people(count):
    pts = []

    for _ in range(count):
        level = np.random.choice(LEVELS)

        x = np.random.uniform(-8.0, 8.0)
        y = np.random.uniform(-2.0, 6.0)

        pts.append([x, y, level])

    return np.array(pts, dtype=float)


agents = spawn_people(AGENT_COUNT)

colors = []
for p in agents:
    band = floor_band(p[2])
    if band == 0:
        colors.append("red")
    elif band == 1:
        colors.append("orange")
    else:
        colors.append("purple")

dots = [Sphere(pos=agents[i], r=0.28, c=colors[i]) for i in range(AGENT_COUNT)]

# -----------------------------
# scene
# -----------------------------
viewer = Plotter(bg="white", axes=1)

floor_meshes = [
    Plane(pos=(0, 0, 0.0), s=(20, 20)).alpha(0.25),
    Plane(pos=(0, 0, LEVEL_GAP), s=(20, 20)).alpha(0.25),
    Plane(pos=(0, 0, 2.0 * LEVEL_GAP), s=(20, 20)).alpha(0.25),
]

door_markers = [Sphere(pos=d, r=0.65, c="green") for d in DOORS]

upper_ramp_vis = Line(
    [UPPER_RAMP_START[0], UPPER_RAMP_START[1], 2.0 * LEVEL_GAP],
    [UPPER_RAMP_END[0], UPPER_RAMP_END[1], LEVEL_GAP]
).lw(8).c("blue")

lower_ramp_vis = Line(
    [LOWER_RAMP_START[0], LOWER_RAMP_START[1], LEVEL_GAP],
    [LOWER_RAMP_END[0], LOWER_RAMP_END[1], 0.0]
).lw(8).c("blue")

viewer.add(*floor_meshes, *door_markers, upper_ramp_vis, lower_ramp_vis, *dots)

# -----------------------------
# animation
# -----------------------------
frame_state = {"n": 0}

def tick(evt):
    global agents, dots, timer_id

    if frame_state["n"] >= MAX_FRAMES:
        viewer.timer_callback("stop", timer_id)
        return

    agents[:] = advance_agents(agents)

    for i in range(AGENT_COUNT):
        dots[i].pos(*agents[i])

    viewer.render()
    frame_state["n"] += 1


viewer.add_callback("timer", tick)
timer_id = viewer.timer_callback("start", dt=35)
viewer.show(interactive=True)
