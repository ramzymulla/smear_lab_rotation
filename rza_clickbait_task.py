import clr
clr.AddReference("OpenCV.Net")
clr.AddReference("System")
from OpenCV.Net import *
import math
import time
import random
import System
from System import Array

# ---------------------------------------------------------------------------
# GridMaze: stores only cell centers (no vertex arrays).
# Hex polygon drawing is removed entirely for performance.
# Cell lookup is now O(1) via axial-coordinate math instead of O(N) loop.
# ---------------------------------------------------------------------------
class GridMaze:
    def __init__(self, maze_bounds, grid_radius, scale_factor):
        self.bounds = maze_bounds
        self.radius = grid_radius
        self.cells = []

        N = grid_radius
        w_ratio = maze_bounds[0] / (math.sqrt(3) * (2 * N + 1)) if N > 0 else maze_bounds[0]
        h_ratio = maze_bounds[1] / (1.5 * 2 * N + 2) if N > 0 else maze_bounds[1]
        self.size = min(w_ratio, h_ratio) * scale_factor

        self.center_x = maze_bounds[0] / 2.0
        self.center_y = maze_bounds[1] / 2.0

        # Build a (q, r) -> index lookup table for O(1) cell resolution
        self._coord_to_idx = {}

        for q in range(-N, N + 1):
            for r in range(-N, N + 1):
                if abs(q + r) <= N:
                    px = self.center_x + self.size * math.sqrt(3) * (q + r / 2.0)
                    py = self.center_y + self.size * 1.5 * r
                    idx = len(self.cells)
                    self.cells.append({
                        'q': q,
                        'r': r,
                        'center': Point(int(px), int(py)),
                        'center_x': px,
                        'center_y': py,
                    })
                    self._coord_to_idx[(q, r)] = idx

        # Precompute circle radius for cell visualization (cheaper than polygons)
        self.viz_radius = int(self.size * 0.45)

    def pixel_to_axial(self, px, py):
        """
        Convert a pixel coordinate to the nearest hex axial (q, r) using
        direct inverse-transform math — O(1), no loop over all cells needed.
        Returns (q, r, cell_index) or (-1, -1, -1) if outside the grid.
        """
        dx = px - self.center_x
        dy = py - self.center_y
        s = self.size

        # Inverse of the flat→pixel transform used in __init__
        q_frac = (dx * math.sqrt(3) / 3.0 - dy / 3.0) / s
        r_frac = dy * 2.0 / 3.0 / s

        # Axial cube-coordinate rounding
        x_frac = q_frac
        z_frac = r_frac
        y_frac = -x_frac - z_frac

        rx = round(x_frac)
        ry = round(y_frac)
        rz = round(z_frac)

        x_diff = abs(rx - x_frac)
        y_diff = abs(ry - y_frac)
        z_diff = abs(rz - z_frac)

        if x_diff > y_diff and x_diff > z_diff:
            rx = -ry - rz
        elif y_diff > z_diff:
            ry = -rx - rz
        else:
            rz = -rx - ry

        q, r = int(rx), int(rz)

        if abs(q) > self.radius or abs(r) > self.radius or abs(q + r) > self.radius:
            return -1, -1, -1

        idx = self._coord_to_idx.get((q, r), -1)
        return q, r, idx


def get_image_shape(img):
    size = img.Size
    return [size.Width, size.Height]


def create_blank_canvas(width, height, channels=3, color=(0, 0, 0)):
    depth = IplDepth.U8
    img = IplImage(Size(width, height), depth, channels)
    fill_color = Scalar.All(color[0]) if channels == 1 else Scalar.Rgb(color[0], color[1], color[2])
    img.Set(fill_color)
    return img


# ---------------------------------------------------------------------------
# Visualization: circles instead of filled polygons.
# Much faster: no overlay allocation, no AddWeighted blend, no vertex arrays.
# ---------------------------------------------------------------------------

def draw_targets_fast(active_target, target_queue, grid, img, target_distribution):
    """
    Draws future targets as proportionally-sized circles and the active
    target as a filled circle. No overlay image or alpha blend needed.
    """
    # Precount future targets per cell (same logic, but applied to circles)
    if target_queue:
        cell_counts = {}
        for cell_idx in target_queue:
            cell_counts[cell_idx] = cell_counts.get(cell_idx, 0) + 1

        min_c = 1
        max_c = 5
        base_i = 50
        range_i = 205
        r = grid.viz_radius

        for cell_idx, count in cell_counts.items():
            if cell_idx < len(grid.cells):
                clamped = max(min_c, min(count, max_c))
                norm = (clamped - min_c) / float(max_c - min_c)
                intensity = base_i + int(norm * range_i)
                color = Scalar.Rgb(0, intensity, intensity)
                center = grid.cells[cell_idx]['center']
                CV.Circle(img, center, r, color, -1)

    # Active target: filled white circle
    if active_target is not None and active_target < len(grid.cells):
        center = grid.cells[active_target]['center']
        CV.Circle(img, center, grid.viz_radius, target_color, -1)


def get_grid_location_fast(grid, centroid_x, centroid_y, active_target, img, target_radius):
    """
    O(1) cell lookup via axial math. Draws a red circle at the mouse cell
    and the target ring/line if a target is active.
    """
    q, r, closest_idx = grid.pixel_to_axial(centroid_x, centroid_y)

    if closest_idx != -1:
        cell = grid.cells[closest_idx]
        CV.Circle(img, cell['center'], grid.viz_radius, mouse_loc_color, 2)

    target_found = False

    if active_target is not None and 0 <= active_target < len(grid.cells):
        target_cell = grid.cells[active_target]
        tx = target_cell['center_x']
        ty = target_cell['center_y']

        dist = math.sqrt((centroid_x - tx) ** 2 + (centroid_y - ty) ** 2)

        CV.Circle(img, Point(int(tx), int(ty)), int(target_radius), threshold_color, thickness=4)
        CV.Line(img, Point(int(tx), int(ty)), Point(int(centroid_x), int(centroid_y)), distance_line_color, thickness=3)

        if dist <= target_radius:
            target_found = True

    return q, r, target_found


def generate_targets(grid_radius, max_targets_per_cell=5, shuffle=True):
    N = grid_radius
    num_cells = 3 * N * (N + 1) + 1
    base_distribution = [0] * num_cells

    sigma = N / 2.0 if N > 0 else 1.0

    idx = 0
    for q in range(-N, N + 1):
        for r in range(-N, N + 1):
            if abs(q + r) <= N:
                hex_dist = math.sqrt((q + r / 2.0) ** 2 * 3 + (r * 1.5) ** 2)
                prob = math.exp(-0.5 * (hex_dist / sigma) ** 2)
                base_distribution[idx] = prob
                idx += 1

    total_prob = sum(base_distribution)
    base_distribution = [p / float(total_prob) for p in base_distribution]
    max_prob = max(base_distribution)

    target_counts = {}
    target_queue = []

    for i, prob in enumerate(base_distribution):
        scaled = 1 + int((prob / max_prob) * (max_targets_per_cell - 1))
        target_counts[i] = scaled
        for _ in range(scaled):
            target_queue.append(i)

    if shuffle:
        random.shuffle(target_queue)

    active_target = None
    if target_queue:
        active_target = target_queue[0]
        target_queue = target_queue[1:]

    return grid_radius, target_queue, base_distribution, active_target


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

grid_radius = 8
max_targets_per_cell = 10
scale_factor = 0.75

_, target_queue, target_distribution, active_target = generate_targets(grid_radius, max_targets_per_cell)

trial_count = 0
reward_left_count = 0
reward_right_count = 0
reward_state = False        # False = hunting target, True = reward available
click = False
click_start_time = 0
drinking = False
reward_left = False
reward_right = False
reward_left_start_time = 0
reward_right_start_time = 0

iti_start_time = 0
iti_duration = 0
in_iti = False
withdrawal_start_time = 0
in_withdrawal_period = False
prev_poke_left = False
prev_poke_right = False

# Grid/canvas cache — rebuilt only on image-dimension change
cached_grid = None
cached_img_dims = None
cached_canvas = None        # Reuse the same IplImage allocation each frame

# Visualization parameters
centroid_color = Scalar.Rgb(255, 255, 255)
mouse_loc_color = Scalar.Rgb(255, 0, 0)
target_color = Scalar.Rgb(255, 255, 255)
grid_color = Scalar.Rgb(128, 128, 128)
centroid_radius = 5
threshold_color = Scalar.Rgb(0, 255, 0)
distance_line_color = Scalar.Rgb(255, 255, 0)
black = Scalar.All(0)


@returns(tuple)
def process(value):
    global trial_count, reward_left_count, reward_right_count
    global target_queue, active_target, target_distribution
    global reward_state, click, click_start_time
    global drinking, reward_left, reward_right
    global reward_left_start_time, reward_right_start_time
    global iti_start_time, iti_duration, in_iti
    global withdrawal_start_time, in_withdrawal_period
    global prev_poke_left, prev_poke_right
    global grid_radius
    global cached_grid, cached_img_dims, cached_canvas

    current_time = time.time()
    reward_duration_left  = 0.05
    reward_duration_right = 0.056
    click_duration        = 0.1
    iti_duration_min      = 1.0
    iti_duration_max      = 5.0
    withdrawal_duration   = 0.5

    target_found_this_frame = False

    centroid_x, centroid_y, image, target_radius = (
        value[0].Item1, value[0].Item2, value[0].Item3, value[0].Item4
    )
    poke_left  = bool(value[1][0])
    poke_right = bool(value[1][1])

    # ------------------------------------------------------------------
    # Grid / canvas cache: rebuild only when image dimensions change.
    # Reuse the same canvas allocation — just clear it each frame.
    # ------------------------------------------------------------------
    img_dims = get_image_shape(image)
    if img_dims != cached_img_dims:
        cached_grid     = GridMaze(img_dims, grid_radius, scale_factor)
        cached_img_dims = img_dims
        cached_canvas   = create_blank_canvas(img_dims[0], img_dims[1])
    else:
        cached_canvas.Set(black)   # Clear without reallocating

    grid   = cached_grid
    canvas = cached_canvas

    # ------------------------------------------------------------------
    # Draw future targets + active target (circles, no overlay blend)
    # ------------------------------------------------------------------
    draw_targets_fast(active_target, target_queue, grid, canvas, target_distribution)

    # ------------------------------------------------------------------
    # Mouse position and target detection
    # ------------------------------------------------------------------
    grid_loc_q, grid_loc_r = None, None

    if not (math.isnan(centroid_x) or math.isnan(centroid_y)):
        grid_loc_q, grid_loc_r, target_found_this_frame = get_grid_location_fast(
            grid, centroid_x, centroid_y, active_target, canvas, target_radius
        )
        CV.Circle(canvas, Point(int(centroid_x), int(centroid_y)), centroid_radius, centroid_color, -1)

        if target_found_this_frame and active_target is not None and not reward_state:
            active_target = None
            reward_state  = True
            click         = True
            click_start_time = current_time

    # ------------------------------------------------------------------
    # State machine: ITI → withdrawal → reward
    # ------------------------------------------------------------------
    if in_iti:
        if current_time - iti_start_time >= iti_duration:
            trial_count += 1
            in_iti = False
            if active_target is None and target_queue:
                active_target  = target_queue[0]
                target_queue   = target_queue[1:]
            # If queue is empty, active_target stays None — session complete.

    elif in_withdrawal_period:
        if not (poke_left or poke_right):
            if current_time - withdrawal_start_time >= withdrawal_duration:
                in_withdrawal_period = False
                in_iti               = True
                iti_start_time       = current_time
                iti_duration         = random.uniform(iti_duration_min, iti_duration_max)
        else:
            withdrawal_start_time = current_time

    elif reward_state:
        if reward_left and current_time - reward_left_start_time >= reward_duration_left:
            reward_left          = False
            in_withdrawal_period = True
            withdrawal_start_time = current_time
            reward_state         = False
        elif reward_right and current_time - reward_right_start_time >= reward_duration_right:
            reward_right         = False
            in_withdrawal_period = True
            withdrawal_start_time = current_time
            reward_state         = False
        elif poke_left and not reward_left and not reward_right:
            reward_left           = True
            reward_left_count    += 1
            reward_left_start_time = current_time
        elif poke_right and not reward_right and not reward_left:
            reward_right          = True
            reward_right_count   += 1
            reward_right_start_time = current_time

    if click and current_time - click_start_time >= click_duration:
        click = False

    prev_poke_left  = poke_left
    prev_poke_right = poke_right
    drinking = reward_state and (poke_left or poke_right)

    return (canvas, Point(centroid_x, centroid_y), reward_state, reward_left, reward_right,
            poke_left, poke_right, drinking, in_iti, click, active_target,
            trial_count, grid_loc_q, grid_loc_r,  
            reward_left_count, reward_right_count, tuple(target_distribution))