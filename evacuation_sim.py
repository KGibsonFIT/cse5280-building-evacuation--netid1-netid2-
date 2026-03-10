import numpy as np
from vedo import Plotter, Plane, Sphere, Line

np.random.seed(7)

# basic setup values
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

# two exit doors on the ground floor
DOORS = np.array([
    [7.0, -4.5, 0.0],
    [-7.0, -4.5, 0.0]
], dtype=float)

# ramps that connect the floors
UPPER_RAMP_START = np.array([-3.0, 3.0], dtype=float)
UPPER_RAMP_END   = np.array([-6.0, -2.0], dtype=float)

LOWER_RAMP_START = np.array([3.0, 3.0], dtype=float)
LOWER_RAMP_END   = np.array([6.0, -2.0], dtype=float)


def clamp01(val):
    # keeps number between 0 and 1
    return max(0.0, min(1.0, val))


def dist_to_line_segment(pt, a, b):
    # distance from a point to a line segment
    seg = b - a
    denom = np.dot(seg, seg)

    if denom == 0:
        return np.linalg.norm(pt - a)

    u = clamp01(np.dot(pt - a, seg) / denom)
    nearest = a + u * seg
    return np.linalg.norm(pt - nearest)


def lerp_height(pt_xy, a, b, z_start, z_end):
    # finds what height the ramp should be at this x,y spot
    seg = b - a
    denom = np.dot(seg, seg)

    if denom == 0:
        return z_start

    u = clamp01(np.dot(pt_xy - a, seg) / denom)
    return z_start + u * (z_end - z_start)


def on_ramp(pt_xy, ramp_a, ramp_b):
    # checks if someone is close enough to be considered on the ramp
    return dist_to_line_segment(pt_xy, ramp_a, ramp_b) <= RAMP_RADIUS


def floor_band(z):
    # tells us what floor someone is on
    if z > 1.5 * LEVEL_GAP:
        return 2
    if z > 0.5 * LEVEL_GAP:
        return 1
    return 0


def target_surface_height(x, y, previous_z):
    # keeps people snapped to floors or ramps
    q = np.array([x, y], dtype=float)

    if previous_z > 1.5 * LEVEL_GAP:
        # top floor people can only go down the upper ramp
        if on_ramp(q, UPPER_RAMP_START, UPPER_RAMP_END):
            return lerp_height(q, UPPER_RAMP_START, UPPER_RAMP_END, 2.0 * LEVEL_GAP, LEVEL_GAP)
        return 2.0 * LEVEL_GAP

    if previous_z > 0.5 * LEVEL_GAP:
        # middle floor uses the lower ramp
        if on_ramp(q, LOWER_RAMP_START, LOWER_RAMP_END):
            return lerp_height(q, LOWER_RAMP_START, LOWER_RAMP_END, LEVEL_GAP, 0.0)
        return LEVEL_GAP

    # ground floor
    return 0.0


def pick_waypoint(agent):
    # decides where the agent should move next
    x, y, z = agent
    xy = agent[:2]

    if z > 1.5 * LEVEL_GAP:
        # top floor heads to upper ramp
        if on_ramp(xy, UPPER_RAMP_START, UPPER_RAMP_END):
            return np.array([UPPER_RAMP_END[0], UPPER_RAMP_END[1], LEVEL_GAP], dtype=float)

        return np.array([UPPER_RAMP_START[0], UPPER_RAMP_START[1], 2.0 * LEVEL_GAP], dtype=float)

    if z > 0.5 * LEVEL_GAP:
        # middle floor heads to lower ramp
        if on_ramp(xy, LOWER_RAMP_START, LOWER_RAMP_END):
            return np.array([LOWER_RAMP_END[0], LOWER_RAMP_END[1], 0.0], dtype=float)

        return np.array([LOWER_RAMP_START[0], LOWER_RAMP_START[1], LEVEL_GAP], dtype=float)

    # ground floor goes to the closest door
    door_dists = np.linalg.norm(DOORS - agent, axis=1)
    return DOORS[np.argmin(door_dists)]


def desired_motion(agent):
    # unit vector pointing toward the current target
    target = pick_waypoint(agent)

    vec = target - agent
    vec[2] = 0.0

    mag = np.linalg.norm(vec)

    if mag < 1e-8:
        return np.zeros(3)

    return vec / mag


def separation_push(idx, crowd):
    # pushes people away if they get too close
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
    # slows people down when they bunch up on ramps
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


def advance_agents(crowd):
    # moves everyone one simulation step
    updated = crowd.copy()

    for i, person in enumerate(crowd):

        nav = desired_motion(person)
        sep = separation_push(i, crowd)
        slow = queue_slowdown(i, crowd)

        velocity = slow * (MOVE_GAIN * nav + AVOID_GAIN * sep)

        proposal = person + TIME_STEP * velocity

        # gently snap to the right floor or ramp
        surf_z = target_surface_height(proposal[0], proposal[1], person[2])

        proposal[2] = person[2] + HEIGHT_GAIN * (surf_z - person[2])

        updated[i] = proposal

    return updated


def spawn_people(count):
    # randomly place people in the building
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


# spheres that represent the people
dots = [Sphere(pos=agents[i], r=0.28, c=colors[i]) for i in range(AGENT_COUNT)]


viewer = Plotter(bg="white", axes=1)

# transparent planes for each floor
floor_meshes = [
    Plane(pos=(0, 0, 0.0), s=(20, 20)).alpha(0.25),
    Plane(pos=(0, 0, LEVEL_GAP), s=(20, 20)).alpha(0.25),
    Plane(pos=(0, 0, 2.0 * LEVEL_GAP), s=(20, 20)).alpha(0.25),
]

# green spheres mark the exits
door_markers = [Sphere(pos=d, r=0.65, c="green") for d in DOORS]

# blue lines show the ramps
upper_ramp_vis = Line(
    [UPPER_RAMP_START[0], UPPER_RAMP_START[1], 2.0 * LEVEL_GAP],
    [UPPER_RAMP_END[0], UPPER_RAMP_END[1], LEVEL_GAP]
).lw(8).c("blue")

lower_ramp_vis = Line(
    [LOWER_RAMP_START[0], LOWER_RAMP_START[1], LEVEL_GAP],
    [LOWER_RAMP_END[0], LOWER_RAMP_END[1], 0.0]
).lw(8).c("blue")


viewer.add(*floor_meshes, *door_markers, upper_ramp_vis, lower_ramp_vis, *dots)

frame_state = {"n": 0}


def tick(evt):
    global agents, dots, timer_id

    # stop when we hit the max frame count
    if frame_state["n"] >= MAX_FRAMES:
        viewer.timer_callback("stop", timer_id)
        return

    # update positions
    agents[:] = advance_agents(agents)

    for i in range(AGENT_COUNT):
        dots[i].pos(*agents[i])

    viewer.render()

    frame_state["n"] += 1


viewer.add_callback("timer", tick)

timer_id = viewer.timer_callback("start", dt=35)

viewer.show(interactive=True)