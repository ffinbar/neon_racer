
import hashlib
import pygame
from pygame.math import Vector2
import shutil
import subprocess

# ============================================================================
# Geometry Utilities
# ============================================================================

def ccw(A, B, C):
    """
    Check if three points are listed in counter-clockwise order.
    
    Args:
        A, B, C: Objects with .x and .y attributes (like Vector2)
    
    Returns:
        bool: True if points are in counter-clockwise order
    """
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)


def line_intersect(A, B, C, D):
    """
    Check if line segment AB intersects with line segment CD.
    
    Args:
        A, B, C, D: Points with .x and .y attributes (like Vector2)
    
    Returns:
        bool: True if the line segments intersect
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


# ============================================================================
# Bezier Curve Utilities
# ============================================================================

def bezier_point(t, p0, p1, p2, p3):
    """
    Compute a point on a cubic bezier curve at parameter t.
    
    Args:
        t: Parameter value [0, 1]
        p0, p1, p2, p3: Control points as (x, y) tuples
    
    Returns:
        tuple: (x, y) coordinates on the curve
    
    Example:
        >>> p = bezier_point(0.5, (0, 0), (1, 2), (3, 2), (4, 0))
        >>> # Returns point at t=0.5 on the curve
    """
    u = 1 - t
    return (
        u**3 * p0[0] + 3*u**2*t * p1[0] + 3*u*t**2 * p2[0] + t**3 * p3[0],
        u**3 * p0[1] + 3*u**2*t * p1[1] + 3*u*t**2 * p2[1] + t**3 * p3[1]
    )


def bezier_curve_points(p0, p1, p2, p3, steps=20):
    """
    Generate points along a cubic bezier curve.
    
    Args:
        p0, p1, p2, p3: Control points as (x, y) tuples
        steps: Number of segments (higher = smoother curve)
    
    Returns:
        list: List of (x, y) tuples representing points on the curve
    
    Example:
        >>> points = bezier_curve_points((0, 0), (1, 2), (3, 2), (4, 0), steps=10)
        >>> # Returns 11 points along the curve
    """
    return [bezier_point(t / steps, p0, p1, p2, p3) for t in range(steps + 1)]


def quadratic_bezier_point(t, p0, p1, p2):
    """
    Compute a point on a quadratic bezier curve at parameter t.
    
    Args:
        t: Parameter value [0, 1]
        p0, p1, p2: Control points as (x, y) tuples
    
    Returns:
        tuple: (x, y) coordinates on the curve
    """
    u = 1 - t
    return (
        u**2 * p0[0] + 2*u*t * p1[0] + t**2 * p2[0],
        u**2 * p0[1] + 2*u*t * p1[1] + t**2 * p2[1]
    )


def quadratic_bezier_points(p0, p1, p2, steps=20):
    """
    Generate points along a quadratic bezier curve.
    
    Args:
        p0, p1, p2: Control points as (x, y) tuples
        steps: Number of segments (higher = smoother curve)
    
    Returns:
        list: List of (x, y) tuples representing points on the curve
    """
    return [quadratic_bezier_point(t / steps, p0, p1, p2) for t in range(steps + 1)]


# ============================================================================
# Angle Utilities
# ============================================================================

def normalize_angle(angle):
    """
    Normalize an angle to the range [0, 360).
    
    Args:
        angle: Angle in degrees
    
    Returns:
        float: Normalized angle in [0, 360)
    
    Example:
        >>> normalize_angle(450)
        90.0
        >>> normalize_angle(-90)
        270.0
    """
    return angle % 360


def angle_difference(angle1, angle2):
    """
    Calculate the shortest angular difference between two angles.
    
    Args:
        angle1, angle2: Angles in degrees
    
    Returns:
        float: Shortest difference in range [-180, 180]
    
    Example:
        >>> angle_difference(10, 350)
        -20.0  # Shortest path is -20 degrees
        >>> angle_difference(350, 10)
        20.0
    """
    diff = angle2 - angle1
    while diff < -180:
        diff += 360
    while diff > 180:
        diff -= 360
    return diff


# ============================================================================
# Math Utilities
# ============================================================================

def clamp(value, min_val, max_val):
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        float: Clamped value
    
    Example:
        >>> clamp(150, 0, 100)
        100
        >>> clamp(-10, 0, 100)
        0
        >>> clamp(50, 0, 100)
        50
    """
    return max(min_val, min(max_val, value))


def lerp(a, b, t):
    """
    Linear interpolation between two values.
    
    Args:
        a: Start value
        b: End value
        t: Interpolation factor [0, 1]
    
    Returns:
        float: Interpolated value
    
    Example:
        >>> lerp(0, 100, 0.5)
        50.0
        >>> lerp(10, 20, 0.25)
        12.5
    """
    return a + (b - a) * t


# ============================================================================
# Process Utilities
# ============================================================================

def is_process_running(patterns):
    """
    Check if a process matching any of the provided patterns is currently running.

    patterns: iterable of strings (regular substrings to search for in the process list)

    Returns True if any matching process is found, False otherwise.
    """
    # Try pgrep -f for fast platform-native checking
    pgrep = shutil.which('pgrep')
    if pgrep:
        for pat in patterns:
            try:
                res = subprocess.run([pgrep, '-f', pat], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if res.returncode == 0:
                    return True
            except Exception:
                pass
        return False

    # Fallback to parsing ps aux output
    try:
        res = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        out = res.stdout
        for pat in patterns:
            if pat in out:
                return True
    except Exception:
        pass
    return False


def quantize(value, step, decimals=2):
    """
    Quantize a value to the nearest multiple of step.
    
    Args:
        value: Value to quantize
        step: Quantization step size
        decimals: Number of decimal places to round to
    
    Returns:
        float: Quantized value
    
    Example:
        >>> quantize(0.847, 0.1)
        0.8
        >>> quantize(1.234, 0.05, decimals=3)
        1.25
    """
    return round(round(value / step) * step, decimals)


# ============================================================================
# Color Utilities
# ============================================================================

def generate_color_from_string(name, saturation=90, value=100):
    """
    Generate a consistent color from a string using MD5 hashing.
    
    Args:
        name: String to generate color from
        saturation: HSV saturation value (0-100)
        value: HSV value/brightness (0-100)
    
    Returns:
        tuple: RGB color as (r, g, b) where each component is 0-255
    
    Example:
        >>> color = generate_color_from_string("Player1")
        >>> # Returns consistent color for "Player1" every time
    
    Note:
        Requires pygame to be installed for Color conversion.
    """
    name_hash = hashlib.md5(name.encode()).hexdigest()
    r = int(name_hash[0:2], 16)
    g = int(name_hash[2:4], 16)
    b = int(name_hash[4:6], 16)
    c = pygame.Color(r, g, b)
    h, s, v, a = c.hsva
    c.hsva = (h, saturation, value, 100)
    return (c.r, c.g, c.b)



# ============================================================================
# AI/Neural Network Utilities
# ============================================================================

def parse_brake_output(output_value, threshold=0.5):
    """
    Parse neural network brake output to boolean.
    
    Args:
        output_value: Raw output from neural network (typically -1 to 1 or 0 to 1)
        threshold: Threshold above which brake is considered active
    
    Returns:
        bool: True if brake should be applied, False otherwise
    
    Example:
        >>> parse_brake_output(0.7)
        True
        >>> parse_brake_output(0.3)
        False
    """
    return True if output_value > threshold else False


def check_gate_crossing(car, gate, track_width=None, track_height=None):
    """
    Check if a car has crossed a gate/checkpoint line.
    
    Args:
        car: Car object with pos, prev_pos attributes
        gate: Gate dict with 'x', 'y', 'angle', 'width' keys
        track_width: Optional track width for toroidal wrapping
        track_height: Optional track height for toroidal wrapping
    
    Returns:
        bool: True if the car crossed the gate line
    
    Example:
        >>> gate = {'x': 100, 'y': 200, 'angle': 0, 'width': 50}
        >>> crossed = check_gate_crossing(car, gate)
    """
    
    gate_pos = Vector2(gate['x'], gate['y'])
    gate_angle = gate.get('angle', 0)
    gate_width = gate.get('width', 100)
    gate_vec = Vector2(1, 0).rotate(gate_angle + 90)
    
    p1 = gate_pos + gate_vec * (gate_width / 2)
    p2 = gate_pos - gate_vec * (gate_width / 2)
    
    if not hasattr(car, 'prev_pos'):
        return False
    
    p3 = car.prev_pos
    p4 = car.pos
    
    # Handle toroidal wrapping if track dimensions provided
    if track_width is not None and track_height is not None:
        mp3 = Vector2(p3.x % track_width, p3.y % track_height)
        mp4 = Vector2(p4.x % track_width, p4.y % track_height)
        
        # Sanity check: if teleport distance is too large, skip
        if mp3.distance_to(mp4) > 100:
            return False
        
        return line_intersect(p1, p2, mp3, mp4)
    else:
        return line_intersect(p1, p2, p3, p4)


def render_text_with_outline(font, text, fg_color, outline_color=(0, 0, 0), outline_width=1, aa=True):
    """Render text with a solid outline and return a Surface.

    Uses multiple blits of the text rendered in the outline_color offset by
    up to `outline_width` pixels, then blits the main text on top.
    """
    # Base rendered surfaces
    base = font.render(text, aa, fg_color)
    outline_surf = font.render(text, aa, outline_color)

    ow = max(0, int(outline_width))
    w = base.get_width() + ow * 2
    h = base.get_height() + ow * 2

    surf = pygame.Surface((w, h), pygame.SRCALPHA)

    # Blit outline copies around the main text
    for ox in range(-ow, ow + 1):
        for oy in range(-ow, ow + 1):
            if ox == 0 and oy == 0:
                continue
            surf.blit(outline_surf, (ox + ow, oy + ow))

    # Blit the main text centered within the outline padding
    surf.blit(base, (ow, ow))
    return surf
