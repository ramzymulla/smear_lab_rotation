import clr
clr.AddReference("OpenCV.Net")
clr.AddReference("System")
from OpenCV.Net import *
import math
import time
import random
import System
from System import Array

# Class to generate hexagonal maze coordinates
class GridMaze:
    def __init__(self, maze_bounds, grid_radius, scale_factor):
        self.bounds = maze_bounds
        self.radius = grid_radius
        self.cells = []
        
        N = grid_radius
        # Calculate optimal hex size to fit image bounds
        w_ratio = maze_bounds[0] / (math.sqrt(3) * (2 * N + 1)) if N > 0 else maze_bounds[0]
        h_ratio = maze_bounds[1] / (1.5 * 2 * N + 2) if N > 0 else maze_bounds[1]
        self.size = min(w_ratio, h_ratio) * scale_factor   # scale size border margin
        
        center_x = maze_bounds[0] / 2.0
        center_y = maze_bounds[1] / 2.0
        
        # Generate hex grid using axial coordinates (q, r)
        for q in range(-N, N + 1):
            for r in range(-N, N + 1):
                if abs(q + r) <= N:
                    px = center_x + self.size * math.sqrt(3) * (q + r / 2.0)
                    py = center_y + self.size * 1.5 * r
                    
                    # Generate pointy-topped hexagon vertices
                    verts = []
                    for i in range(6):
                        angle_deg = 60 * i - 30
                        angle_rad = math.pi / 180.0 * angle_deg
                        vx = px + self.size * math.cos(angle_rad)
                        vy = py + self.size * math.sin(angle_rad)
                        verts.append(Point(int(vx), int(vy)))
                    
                    arr = Array.CreateInstance(Point, 6)
                    for i, v in enumerate(verts): 
                        arr[i] = v
                    
                    self.cells.append({
                        'q': q,
                        'r': r,
                        'center': Point(int(px), int(py)),
                        'vertices': arr
                    })

def draw_hex_outline(img, vertices, color, thickness):
    for i in range(6):
        CV.Line(img, vertices[i], vertices[(i+1)%6], color, thickness)

def draw_grid(grid, img):
    for cell in grid.cells:
        draw_hex_outline(img, cell['vertices'], grid_color, thickness=2)

def get_image_shape(img):
    size = img.Size
    return [size.Width, size.Height]

def draw_target_distribution(target_distribution, grid, img, max_intensity=255):
    if sum(target_distribution) > 0:
        max_prob = max(target_distribution)
    else:
        max_prob = 1.0
    
    overlay = create_blank_canvas(img.Size.Width, img.Size.Height)
    
    for i, prob in enumerate(target_distribution):
        if prob > 0 and i < len(grid.cells):
            norm_prob = prob / float(max_prob)
            intensity = int(norm_prob * max_intensity)
            dist_color = Scalar.Rgb(intensity, 0, 0)
            
            cell = grid.cells[i]
            CV.FillConvexPoly(overlay, cell['vertices'], dist_color)
    
    alpha = 0.5
    CV.AddWeighted(img, 1.0, overlay, alpha, 0.0, img)
    
    return img

global initial_cell_counts
global max_initial_count
initial_cell_counts = {}
max_initial_count = 1

def draw_future_targets(target_queue, grid, img):
    all_targets = list(target_queue)

    if not all_targets:
        return img

    overlay = create_blank_canvas(img.Size.Width, img.Size.Height)

    cell_counts = {}
    for cell_idx in all_targets:
        if cell_idx in cell_counts:
            cell_counts[cell_idx] += 1
        else:
            cell_counts[cell_idx] = 1

    min_possible_count = 1
    max_possible_count = 5
    
    base_intensity = 50
    intensity_range = 205
    
    for cell_idx, count in cell_counts.items():
        if cell_idx < len(grid.cells):
            clamped_count = max(min_possible_count, min(count, max_possible_count))
            normalized_position = (clamped_count - min_possible_count) / float(max_possible_count - min_possible_count)
            intensity = base_intensity + int(normalized_position * intensity_range)
            
            target_future_color = Scalar.Rgb(0, intensity, intensity)

            cell = grid.cells[cell_idx]
            CV.FillConvexPoly(overlay, cell['vertices'], target_future_color)

    alpha = 0.5
    CV.AddWeighted(img, 1.0, overlay, alpha, 0.0, img)

    return img

def get_grid_location(grid, centroid_x, centroid_y, active_target, img, target_radius):
    closest_dist = 1000000.0
    closest_idx = -1
    
    for i, cell in enumerate(grid.cells):
        dist = math.sqrt((centroid_x - cell['center'].X)**2 + (centroid_y - cell['center'].Y)**2)
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = i

    q, r = 0, 0
    if closest_idx != -1 and closest_dist <= grid.size:
        cell = grid.cells[closest_idx]
        CV.FillConvexPoly(img, cell['vertices'], mouse_loc_color)
        q = cell['q']
        r = cell['r']
            
    target_found = False
    
    if active_target is not None and 0 <= active_target < len(grid.cells):
        target_cell = grid.cells[active_target]
        center_x = target_cell['center'].X
        center_y = target_cell['center'].Y
        
        dist = math.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2)
        
        CV.Circle(img, Point(int(center_x), int(center_y)), int(target_radius), threshold_color, thickness=4)
        CV.Line(img, Point(int(center_x), int(center_y)), Point(int(centroid_x), int(centroid_y)), distance_line_color, thickness=3)
        
        if dist <= target_radius:
            target_found = True
            
    return q, r, target_found

def create_blank_canvas(width, height, channels=3, color=(0, 0, 0)):
    depth = IplDepth.U8
    img = IplImage(Size(width, height), depth, channels)
    if channels == 1:
        fill_color = Scalar.All(color[0])
    else:
        fill_color = Scalar.Rgb(color[0], color[1], color[2])
    img.Set(fill_color)
    
    return img

def generate_targets(grid_radius, max_targets_per_cell=5, shuffle=True):
    N = grid_radius
    num_cells = 3 * N * (N + 1) + 1
    base_distribution = [0] * num_cells
    
    sigma = N / 2.0 if N > 0 else 1.0
    
    idx = 0
    for q in range(-N, N + 1):
        for r in range(-N, N + 1):
            if abs(q + r) <= N:
                hex_dist = math.sqrt((q + r / 2.0)**2 * 3 + (r * 1.5)**2)
                prob = math.exp(-0.5 * (hex_dist / sigma) ** 2)
                base_distribution[idx] = prob
                idx += 1
    
    total_prob = sum(base_distribution)
    base_distribution = [p / float(total_prob) for p in base_distribution]
    
    max_prob = max(base_distribution)
    
    target_counts = {}
    total_targets = 0
    
    for i, prob in enumerate(base_distribution):
        scaled_targets = 1 + int((prob / max_prob) * (max_targets_per_cell - 1))
        target_counts[i] = scaled_targets
        total_targets += scaled_targets
    
    target_queue = []
    for cell_idx, count in target_counts.items():
        for _ in range(count):
            target_queue.append(cell_idx)
    
    if shuffle:
        random.shuffle(target_queue)

    active_target = None
    if target_queue:
        active_target = target_queue[0]
        target_queue = target_queue[1:]
    
    global initial_cell_counts
    global max_initial_count
    initial_cell_counts = target_counts.copy()
    max_initial_count = max_targets_per_cell
    
    return grid_radius, target_queue, base_distribution, active_target

def draw_targets(active_target, target_queue, grid, img, draw_distribution=False, draw_future=False):
    if draw_distribution:
        img = draw_target_distribution(target_distribution, grid, img)
    
    if draw_future:
        img = draw_future_targets(target_queue, grid, img)
    
    if active_target is not None and active_target < len(grid.cells):
        cell = grid.cells[active_target]
        CV.FillConvexPoly(img, cell['vertices'], target_color)
    
    return img


global grid_radius
global max_targets_per_cell
global scale_factor
grid_radius = 8
max_targets_per_cell = 10
scale_factor = 0.75

global target_queue
global active_target
global target_distribution

_, target_queue, target_distribution, active_target = generate_targets(grid_radius, max_targets_per_cell)

global trial_count
global reward_left_count
global reward_right_count
global reward_state
global click
global click_start_time
global drinking
global reward_left
global reward_right
global reward_left_start_time
global reward_right_start_time

trial_count = 0
reward_left_count = 0
reward_right_count = 0
reward_state = True
click = False
click_start_time = 0
drinking = False
reward_left = False
reward_right = False
reward_left_start_time = 0
reward_right_start_time = 0

global iti_start_time
global iti_duration
global in_iti
global withdrawal_start_time
global in_withdrawal_period
global prev_poke_left
global prev_poke_right

iti_start_time = 0
iti_duration = 0
in_iti = False
withdrawal_start_time = 0
in_withdrawal_period = False
prev_poke_left = False
prev_poke_right = False

# Visualization parameters
centroid_color = Scalar.Rgb(255, 255, 255)
mouse_loc_color = Scalar.Rgb(255, 0, 0)
target_color = Scalar.Rgb(255, 255, 255)
grid_color = Scalar.Rgb(128, 128, 128)
centroid_radius = 5
threshold_color = Scalar.Rgb(0, 255, 0)
distance_line_color = Scalar.Rgb(255, 255, 0)

# Caching variables for the grid
global cached_grid
global cached_dims
cached_grid = None
cached_dims = None

@returns(tuple)
def process(value):
    global trial_count
    global reward_left_count
    global reward_right_count
    global target_queue
    global active_target
    global target_distribution
    global reward_state
    global click
    global click_start_time
    global drinking
    global reward_left
    global reward_right
    global reward_left_start_time
    global reward_right_start_time
    global iti_start_time
    global iti_duration
    global in_iti
    global withdrawal_start_time
    global in_withdrawal_period
    global prev_poke_left
    global prev_poke_right
    global grid_radius
    
    # Grid caching globals
    global cached_grid
    global cached_dims

    current_time = time.time()
    reward_duration_left = 0.032
    reward_duration_right = 0.032
    click_duration = 0.1
    iti_duration_min = 1.0
    iti_duration_max = 5.0
    withdrawal_duration = 0.5
    
    target_found_this_frame = False

    centroid_x, centroid_y, image, target_radius = value[0].Item1, value[0].Item2, value[0].Item3, value[0].Item4
    poke_left, poke_right = bool(value[1][0]), bool(value[1][1])
    
    grid_loc_q, grid_loc_r = None, None
    img_dims = get_image_shape(image)
    
    # Only calculate the grid once, or if the camera resolution changes
    if cached_grid is None or cached_dims != img_dims:
        cached_grid = GridMaze(img_dims, grid_radius, scale_factor)
        cached_dims = img_dims
        
    grid = cached_grid
    
    canvas = create_blank_canvas(img_dims[0], img_dims[1])

    draw_targets(active_target, target_queue, grid, canvas, draw_distribution=False, draw_future=True)

    if not (math.isnan(centroid_x) or math.isnan(centroid_y)):
        grid_loc_q, grid_loc_r, target_found_this_frame = get_grid_location(grid, centroid_x, centroid_y, active_target, canvas, target_radius)
        CV.Circle(canvas, Point(int(centroid_x), int(centroid_y)), centroid_radius, centroid_color, -1)
        
        if target_found_this_frame and active_target is not None and not reward_state:
            active_target = None
            reward_state = True
            click = True
            click_start_time = current_time

    if in_iti:
        if current_time - iti_start_time >= iti_duration:
            trial_count += 1
            in_iti = False
            
            if active_target is None and target_queue:
                active_target = target_queue[0]
                target_queue = target_queue[1:]
            
    elif in_withdrawal_period:
        if not (poke_left or poke_right):
            if current_time - withdrawal_start_time >= withdrawal_duration:
                in_withdrawal_period = False
                in_iti = True
                iti_start_time = current_time
                iti_duration = random.uniform(iti_duration_min, iti_duration_max)
        else:
            withdrawal_start_time = current_time
            
    elif reward_state:
        if reward_left and current_time - reward_left_start_time >= reward_duration_left:
            reward_left = False
            in_withdrawal_period = True
            withdrawal_start_time = current_time
            reward_state = False
        elif reward_right and current_time - reward_right_start_time >= reward_duration_right:
            reward_right = False
            in_withdrawal_period = True
            withdrawal_start_time = current_time
            reward_state = False
        elif poke_left and not reward_left and not reward_right:
            reward_left = True
            reward_left_count += 1
            reward_left_start_time = current_time
        elif poke_right and not reward_right and not reward_left:
            reward_right = True
            reward_right_count += 1
            reward_right_start_time = current_time

    if click and current_time - click_start_time >= click_duration:
        click = False

    prev_poke_left, prev_poke_right = poke_left, poke_right
    drinking = poke_left or poke_right

    return (canvas, Point(centroid_x, centroid_y), reward_state, reward_left, reward_right, 
            poke_left, poke_right, drinking, in_iti, click, active_target, 
            trial_count, reward_left_count, reward_right_count, tuple(target_distribution))