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
# When n_tiles_per_side == 1, only a single center cell is generated.
# ---------------------------------------------------------------------------
class GridMaze:
    def __init__(self, maze_bounds, n_tiles_per_side, scale_factor):
        self.bounds = maze_bounds
        self.radius = n_tiles_per_side
        self.scale = scale_factor
        self.cells = []

        N = n_tiles_per_side
        w_ratio = maze_bounds[0] / (math.sqrt(3) * (2 * N + 1)) if N > 0 else maze_bounds[0]
        h_ratio = maze_bounds[1] / (1.5 * 2 * N + 2) if N > 0 else maze_bounds[1]
        self.size = min(w_ratio, h_ratio) * scale_factor
        print(self.size)

        self.center_x = maze_bounds[0] / 2.0
        self.center_y = maze_bounds[1] / 2.0

        self._coord_to_idx = {}

        if N == 1:
            # Special case: single center tile only
            self.cells.append({
                'q': 0,
                'r': 0,
                'center': Point(int(self.center_x), int(self.center_y)),
                'center_x': self.center_x,
                'center_y': self.center_y,
            })
            self._coord_to_idx[(0, 0)] = 0
        else:
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

        self.viz_radius = int(self.size * 0.45)

    def pixel_to_axial(self, px, py):
        """
        Convert a pixel coordinate to the nearest hex axial (q, r) using
        direct inverse-transform math — O(1), no loop over all cells needed.
        Returns (q, r, cell_index) or (-1, -1, -1) if outside the grid.
        When n_tiles_per_side == 1, always returns the single center cell.
        """
        if self.radius == 1:
            return 0, 0, 0

        dx = px - self.center_x
        dy = py - self.center_y
        s = self.size

        q_frac = (dx * math.sqrt(3) / 3.0 - dy / 3.0) / s
        r_frac = dy * 2.0 / 3.0 / s

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


def build_future_draw_list(target_queue, grid):
    """
    Precompute the list of CV.Circle arguments for all future targets.
    Called only when the queue changes (i.e. on target hits), not every frame.
    Returns a list of (Point, radius, Scalar) tuples.
    """
    draw_list = []
    if not target_queue:
        return draw_list

    cell_counts = {}
    for cell_idx in target_queue:
        cell_counts[cell_idx] = cell_counts.get(cell_idx, 0) + 1

    min_c   = 1
    max_c   = 5
    base_i  = 50
    range_i = 205
    r       = grid.viz_radius
    n_cells = len(grid.cells)

    for cell_idx, count in cell_counts.items():
        if cell_idx < n_cells:
            clamped   = max(min_c, min(count, max_c))
            norm      = (clamped - min_c) / float(max_c - min_c)
            intensity = base_i + int(norm * range_i)
            draw_list.append((grid.cells[cell_idx]['center'], r, Scalar.Rgb(0, intensity, intensity)))

    return draw_list


def draw_targets_fast(active_target, grid, img, future_draw_list):
    """
    Draws future targets from a precomputed draw list (no per-frame looping
    over the queue) then draws the active target on top.
    """
    for center, r, color in future_draw_list:
        CV.Circle(img, center, r, color, -1)

    if active_target is not None and active_target < len(grid.cells):
        CV.Circle(img, grid.cells[active_target]['center'], grid.viz_radius, target_color, -1)


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
        distFromCenter = math.sqrt((centroid_x - grid.center_x) ** 2 + (centroid_y - grid.center_y) ** 2)
        

        CV.Circle(img, Point(int(tx), int(ty)), int(target_radius), threshold_color, thickness=4)
        CV.Line(img, Point(int(tx), int(ty)), Point(int(centroid_x), int(centroid_y)), distance_line_color, thickness=3)

        if dist <= target_radius:
            target_found = True

    return q, r, target_found


def generate_targets(n_tiles_per_side, max_targets_per_cell=5, shuffle=True):
    N = n_tiles_per_side

    if N == 1:
        # Single cell: all targets go to cell index 0
        target_queue = [0] * max_targets_per_cell*10  # Arbitrary large number of targets for the single cell
        if shuffle:
            random.shuffle(target_queue)
        base_distribution = [1.0]
        active_target = 0 if target_queue else None
        if target_queue:
            target_queue = target_queue[1:]
        return N, target_queue, base_distribution, active_target

    num_cells = 3 * N * (N + 1) + 1
    base_distribution = [0] * num_cells

    sigma = N / 2.0

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

    return N, target_queue, base_distribution, active_target


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

n_tiles_per_side = 8
max_targets_per_cell = 100
scale_factor = 0.5

_, target_queue, target_distribution, active_target = generate_targets(n_tiles_per_side, max_targets_per_cell)

trial_count = 0
reward_left_count = 0
reward_right_count = 0
reward_state = False
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

reward_timeout_duration = 5.0  # seconds animal has to poke after target hit
reward_window_start_time = 0    # timestamp when reward_state became True
failed_trial_count = 0          # incremented on timeout

cached_grid = None
cached_img_dims = None
cached_canvas = None

# Precomputed draw list for draw_targets_fast — rebuilt only when the queue changes.
# Each entry is (center_Point, radius, Scalar_color) ready to pass straight to CV.Circle.
cached_future_draw_list = []
cached_queue_id = None   # id() of the target_queue list at last rebuild

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
    global n_tiles_per_side
    global cached_grid, cached_img_dims, cached_canvas
    global cached_future_draw_list, cached_queue_id
    global reward_window_start_time, failed_trial_count

    current_time = time.time()
    reward_duration_left  = 0.049
    reward_duration_right = 0.05
    click_duration        = 0.1
    iti_duration_min      = 1.0
    iti_duration_max      = 5.0
    withdrawal_duration   = 0.5

    target_found_this_frame = False

    centroid_x, centroid_y, image = (
        value[0].Item1, value[0].Item2, value[0].Item3
    )
    poke_left  = bool(value[1][0])
    poke_right = bool(value[1][1])
    nRewards = reward_left_count + reward_right_count
    target_radius = float(value[1][2])
    if nRewards==0:
        target_radius = 250
    elif n_tiles_per_side == 1:
        target_radius = max(100.0, 250.0 - (nRewards * 3))
    elif 0:
        # target_radius = max(target_radius, 250.0 - (nRewards * 3))
        target_radius = max(target_radius, 100.0 - (nRewards * 2))
    elif 1 and target_radius < 300:
        target_radius = 1
    
        


    # ------------------------------------------------------------------
    # Grid / canvas cache: rebuild only when image dimensions change.
    # ------------------------------------------------------------------
    img_dims = get_image_shape(image)
    if img_dims != cached_img_dims:
        cached_grid     = GridMaze(img_dims, n_tiles_per_side, scale_factor)
        cached_img_dims = img_dims
        cached_canvas   = create_blank_canvas(img_dims[0], img_dims[1])
        cached_queue_id = None  # force draw list rebuild on dimension change
    else:
        cached_canvas.Set(black)

    grid   = cached_grid
    canvas = cached_canvas

    # Rebuild the future draw list only when the queue has changed.
    current_queue_id = id(target_queue)
    if current_queue_id != cached_queue_id:
        cached_future_draw_list = build_future_draw_list(target_queue, grid)
        cached_queue_id = current_queue_id

    draw_targets_fast(active_target, grid, canvas, cached_future_draw_list)

    # ------------------------------------------------------------------
    # Mouse position and target detection
    # ------------------------------------------------------------------
    grid_loc_q, grid_loc_r = None, None

    if not (math.isnan(centroid_x) or math.isnan(centroid_y)):
        grid_loc_q, grid_loc_r, target_found_this_frame = get_grid_location_fast(
            grid, centroid_x, centroid_y, active_target, canvas, target_radius
        )
        CV.Circle(canvas, Point(int(centroid_x), int(centroid_y)), centroid_radius, centroid_color, -1)

        # BUG FIX: guard against re-triggering if we're already in reward state
        # or mid-ITI. Only a clean target hit during the hunt phase counts.
        if target_found_this_frame and active_target is not None and not reward_state and not in_iti and not in_withdrawal_period:
            active_target            = None
            reward_state             = True
            reward_window_start_time = current_time
            click                    = True
            click_start_time         = current_time
            # Queue identity hasn't changed yet (next target comes after ITI),
            # so draw list stays valid showing remaining future targets.

    # ------------------------------------------------------------------
    # State machine: ITI → withdrawal → reward
    # BUG FIX: each branch is strictly exclusive via elif, so reward_state
    # can never process a poke on the same frame a target hit occurred.
    # ------------------------------------------------------------------
    if in_iti:
        if current_time - iti_start_time >= iti_duration:
            trial_count += 1
            in_iti = False
            # BUG FIX: was reading target_queue[0] then separately slicing,
            # which is correct but the cached_queue_id must be invalidated
            # here since target_queue is reassigned via slicing.
            if active_target is None and target_queue:
                active_target   = target_queue[0]
                target_queue    = target_queue[1:]
                cached_queue_id = None  # force draw list rebuild next frame

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
            reward_left           = False
            in_withdrawal_period  = True
            withdrawal_start_time = current_time
            reward_state          = False
        elif reward_right and current_time - reward_right_start_time >= reward_duration_right:
            reward_right          = False
            in_withdrawal_period  = True
            withdrawal_start_time = current_time
            reward_state          = False
        elif poke_left and not reward_left and not reward_right:
            reward_left            = True
            reward_left_count     += 1
            reward_left_start_time = current_time
        elif poke_right and not reward_right and not reward_left:
            reward_right           = True
            reward_right_count    += 1
            reward_right_start_time = current_time
        # Only reached on frames AFTER the target-hit frame, since the
        # target-hit block above sets reward_state=True but this elif
        # won't execute until the next call.
        elif current_time - reward_window_start_time >= reward_timeout_duration:
            # Animal did not poke in time — count as failed trial and
            # skip reward delivery, going straight to ITI.
            failed_trial_count += 1
            reward_state        = False
            reward_left         = False
            reward_right        = False
            in_iti              = True
            iti_start_time      = current_time
            iti_duration        = random.uniform(iti_duration_min, iti_duration_max)

    if click and current_time - click_start_time >= click_duration:
        click = False



    prev_poke_left  = poke_left
    prev_poke_right = poke_right
    drinking = reward_state and (poke_left or poke_right)
    total_trials = trial_count + failed_trial_count

    return (canvas, 
            Point(centroid_x, centroid_y), 
            reward_state, 
            reward_left,
            reward_right,
            poke_left, 
            poke_right, 
            drinking, 
            in_iti, 
            click, 
            active_target,
            grid_loc_q, 
            grid_loc_r,
            target_radius, 
            
            failed_trial_count, 
            trial_count, 
            reward_left_count, 
            reward_right_count, 
            tuple(target_distribution))