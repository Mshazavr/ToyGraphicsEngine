import copy
import math
import sys
import time
from sshkeyboard import listen_keyboard
import numpy as np

EPSILON = 1e-10

class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"
    GRAY_SCALE = [f"\033[38;5;{x}m" for x in range(232, 256)]
    ASCII_SCALE = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'."

    def gray_from_intensity(intensity: float):
        """ Intensity is between -1 and 1. -1 corresponds to the darkest shade"""
        if math.fabs(intensity) > 1 + EPSILON:
            raise ValueError

        num_gray_colors = len(Colors.GRAY_SCALE)
        gray_color_index = min(int((((intensity + 1.0) * 0.5)) * num_gray_colors), num_gray_colors-1)
        return Colors.GRAY_SCALE[gray_color_index]

    def ascii_from_intensity(intensity: float):
        """ Intensity is between -1 and 1. -1 corresponds to the darkest shade"""
        intensity *= -1
        if math.fabs(intensity) > 1 + EPSILON:
            raise ValueError

        num_chars = len(Colors.ASCII_SCALE)
        char_index = min(int((((intensity + 1.0) * 0.5)) * num_chars), num_chars-1)
        return Colors.ASCII_SCALE[char_index]

class Matrix:
    def __init__(self, rows: list[list[float]]):
        self.rows = rows
        self.n = len(rows)
        self.m = len(rows[0])
    
    def __mul__(self, other):
        assert self.m == other.n 
        new_rows = []
        for i in range(self.n):
            new_rows.append([0.0 for _ in range(other.m)])
            for j in range(other.m):
                for k in range(self.m):
                    new_rows[i][j] += self.rows[i][k] * other.rows[k][j]

        return Matrix(new_rows)


class Vector:
    def __init__(self, values: list[float]):
        self.n = len(values)
        self.values = values

    def transform(self, mat: Matrix):
        self_as_matrix = Matrix([self.values])
        transformed_matrix = self_as_matrix * mat
        return Vector(transformed_matrix.rows[0])

    def __add__(self, other):
        return Vector([x + y for x, y in zip(self.values, other.values)])

    def __sub__(self, other):
        return Vector([x - y for x, y in zip(self.values, other.values)])

    def __neg__(self):
        return Vector([-x for x in self.values])

    def __mul__(self, num: float):
        return Vector([x*num for x in self.values])

    def cross(self, other):
        return Vector([
            self.values[1] * other.values[2] - self.values[2] * other.values[1],
            self.values[2] * other.values[0] - self.values[0] * other.values[2],
            self.values[0] * other.values[1] - self.values[1] * other.values[0],
        ])
    
    def norm(self) -> float:
        return math.sqrt(sum(x ** 2 for x in self.values))
    
    def dot(self, other):
        return sum(x * y for x, y in zip(self.values, other.values))

    def scale(self, s: float):
        return Vector([x * s for x in self.values])
    
    def normalized(self):
        norm = self.norm()
        return Vector([x / norm for x in self.values])


def lin_scale_mat(k: float):
    return np.array([
        [k, 0, 0],
        [0, k, 0],
        [0, 0, k],
    ])

def xy_rot_mat(theta: float):
    return np.array([
        [ math.cos(theta), math.sin(theta), 0],
        [-math.sin(theta), math.cos(theta), 0],
        [               0,               0, 1],
    ])

def xz_rot_mat(theta: float):
    return np.array([
        [ math.cos(theta), 0, math.sin(theta)],
        [               0, 1,               0],
        [-math.sin(theta), 0, math.cos(theta)],
    ])

def yz_rot_mat(theta: float):
    return np.array([
        [1,                0,               0],
        [0,  math.cos(theta), math.sin(theta)],
        [0, -math.sin(theta), math.cos(theta)],
    ])



## Triangle functions 
def triangle_norm(vertices: np.ndarray[np.float64]):
    cross_product = np.cross(
        (vertices[1] - vertices[0]),
        (vertices[2] - vertices[1])
    )
    return cross_product / np.linalg.norm(cross_product)

def triangle_area(vertices: np.ndarray[np.float64]):
    return np.linalg.norm(
        np.cross(
            (vertices[1] - vertices[0]),
            (vertices[2] - vertices[1])
        ) 
    ) * 0.5

def triangle_ray_intersection(
    vertices: np.ndarray[np.float64], 
    ray_shift: np.ndarray[np.float64],
    ray_direction: np.ndarray[np.float64]
) -> float:
    # (ls + ld * t) dot norm = pc
    # t * (ld dot norm) = pc - ls dot norm
    norm = triangle_norm(vertices)
    plane_constant = norm.dot(vertices[0])

    denominator = ray_direction.dot(norm)
    if np.abs(denominator) <= EPSILON:
        return None
    
    nominator = plane_constant - ray_shift.dot(norm)

    t = nominator / denominator
    return t
    

## Face functions 
def normal_at_face_point(
    vertices: np.ndarray[np.float64],
    normals: np.ndarray[np.float64], 
    point: np.ndarray[np.float64]
):
    """vertices_and_norms has 6 rows and 3 columns. 3 rows for the vertices. 
    3 rows for the vertex normals (in a corresponding order)
    """
    area1 = triangle_area(np.vstack([vertices[0:2], point]))
    area2 = triangle_area(np.vstack([vertices[1:3], point]))
    area3 = triangle_area(np.vstack([vertices[0], vertices[2], point]))
    area = area1 + area2 + area3
    alpha, beta, gamma = area1 / area, area2 / area, area3 / area
    normal = np.sum(np.multiply(normals, np.array([alpha, beta, gamma])[:, np.newaxis]), axis=0)
    return normal / np.linalg.norm(normal)

def normalize_rows(arr: np.ndarray[np.float64]):
    return np.multiply( 
        arr,
        np.reciprocal(np.linalg.norm(arr, axis=1))[:, np.newaxis]
    )

class Model:
    def __init__(
        self, 
        vertices: np.ndarray[np.float64], 
        normals: np.ndarray[np.float64]
    ):
        """vertices has shape (3*n, 3). Rows 3k, 3k+1 and 3k+2 represent the vertices of the k-th face. 
        normals has the same shape and contains the corresponding vertex normals 
        """
        self.vertices = vertices
        self.normals = normals
        assert vertices.shape[0] % 3 == 0
        self.num_faces = int(self.vertices.shape[0] / 3)

    def transform(self, mat: np.ndarray[np.float64]):
        self.vertices[:] = self.vertices.dot(mat)
        self.normals[:] = normalize_rows(self.normals.dot(mat))

def parse_obj_file(path: str) -> Model:
    vertices = []
    vertex_normals = []
    face_vertices = []
    face_normals = []
    with open(path, 'r') as file:
        for line in file:
            elements = line.split()
            if len(elements) == 0:
                continue
            if elements[0] == 'v':
                coordinates = [float(x) for x in elements[1:]]
                vertices.append(coordinates)
            elif elements[0] == 'vn':
                coordinates = [float(x) for x in elements[1:]]
                vertex_normals.append(coordinates)
            elif elements[0] == 'f':
                index_list = [[int(x) if x != '' else None for x in e.split('/')] for e in elements[1:]]
                for l in index_list:
                    if l[0] > 0:
                        l[0] -= 1 
                    if l[2] > 0:
                        l[2] -= 1
                face_vertices += [vertices[index_list[0][0]], vertices[index_list[1][0]], vertices[index_list[2][0]]]
                face_normals += [vertex_normals[index_list[0][2]], vertex_normals[index_list[1][2]], vertex_normals[index_list[2][2]]]
                if len(index_list) == 4:
                    face_vertices += [vertices[index_list[0][0]], vertices[index_list[3][0]], vertices[index_list[2][0]]]
                    face_normals += [vertex_normals[index_list[0][2]], vertex_normals[index_list[3][2]], vertex_normals[index_list[2][2]]]

    return Model(np.array(face_vertices), np.array(face_normals))


class Window:
    def __init__(
            self, 
            x_pixel_sz: int, 
            y_pixel_sz: int, 
            x_scale: float, 
            y_scale: float,
            camera_position: np.ndarray[np.float64],
            camera_direction: np.ndarray[np.float64],
            camera_x_vector: np.ndarray[np.float64], 
            camera_y_vector: np.ndarray[np.float64],
            focal_length: float,
            light_direction: np.ndarray[np.float64],
            models: dict[str, Model],
        ):
        self.x_pixel_sz = x_pixel_sz
        self.y_pixel_sz = y_pixel_sz
        self.x_scale = x_scale 
        self.y_scale = y_scale 
        self.camera_position = camera_position
        self.camera_direction = camera_direction
        self.camera_x_vector = camera_x_vector
        self.camera_y_vector = camera_y_vector
        self.focal_length = focal_length
        self.light_direction = light_direction
        self.models = models
        self.frame_data = [
            [' ' for _ in range(2*x_pixel_sz+1)] 
            for _ in range(2*y_pixel_sz+1)
        ]
        self.refresh_frame()
        for line in self.frame_data:
            sys.stdout.write(''.join(line) + '\n')

    def clear_frame(self):
        self.frame_data = [
            [' ' for _ in range(2*self.x_pixel_sz+1)] 
            for _ in range(2*self.y_pixel_sz+1)
        ]

    def refresh_pixel(self, r, c):
        x = (c - self.x_pixel_sz) * (self.x_scale / self.x_pixel_sz)
        y = (r - self.y_pixel_sz) * (self.y_scale / self.y_pixel_sz)
        ray_direction = (
            (self.camera_direction * self.focal_length) + 
            (self.camera_x_vector * x) + 
            (self.camera_y_vector * y)
        )
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        top_norm = None
        min_z_value = math.inf

        for model in self.models.values():
            for i in range(model.num_faces):
                z_value = triangle_ray_intersection(
                    model.vertices[3*i:3*(i+1)],
                    self.camera_position, 
                    ray_direction
                )
                if z_value is None:
                    continue

                target_point = self.camera_position + (ray_direction * z_value)
                
                area1 = (
                    triangle_area(np.vstack([target_point, model.vertices[3*i: 3*i+2]])) + 
                    triangle_area(np.vstack([target_point, model.vertices[3*i+1: 3*i+3]])) + 
                    triangle_area(np.vstack([target_point, model.vertices[3*i], model.vertices[3*i+2]]))
                )

                area2 = triangle_area(model.vertices[3*i: 3*(i+1)])

                if np.abs(area1 - area2) > EPSILON:
                    continue

                if z_value is None:
                    continue
                if z_value < min_z_value:
                    min_z_value = z_value
                    top_norm = normal_at_face_point(
                        model.vertices[3*i:3*(i+1)], 
                        model.normals[3*i:3*(i+1)],
                        target_point
                    )

        if top_norm is not None: 
            color = Colors.gray_from_intensity(top_norm.dot(self.light_direction))
            self.frame_data[r][c] = f'{color}#'
        else:
            self.frame_data[r][c] = ' '


    
    def _reset_render(self):
        for _ in range(len(self.frame_data)):
            sys.stdout.write("\033[F")  # Move the cursor up one line
            sys.stdout.write("\033[K")  # Delete line

    
    def refresh_frame(self):
        for r in range(2*self.y_pixel_sz+1):
            for c in range(2*self.x_pixel_sz+1):
                self.refresh_pixel(r, c)

    
    def render_frame(self):
        self._reset_render()
        for line in self.frame_data:
            try:
                line_string = ''.join(line) + '\n'
                line_bytes = line_string.encode('utf-8')
                written = sys.stdout.write(line_string)
            except BlockingIOError:
                time.sleep(0.01)
                sys.stdout.write(line_bytes[written:].decode('utf-8'))
    
    def refresh_and_render(self):
        self.refresh_frame()
        self.render_frame()

    def update_model(self, model_name: str, new_model: Model):
        self.models[model_name] = new_model
        self.refresh_and_render()

    def update_light_direction(self, light_direction: Vector):
        self.light_direction = light_direction
        self.refresh_and_render()


def main():
    #cube = get_cube(7)
    #cube = get_smooth_cube(7)
    #cube = parse_obj_file('cube.obj')

    #cube = parse_obj_file('dodecahedron.obj')
    #cube = cube.transform(lin_scale_mat(10))
    
    #cube = parse_obj_file('arrow_centered.obj')
    #cube.transform(lin_scale_mat(10))

    cube = parse_obj_file('cube.obj')
    cube.transform(lin_scale_mat(20))

    light_direction = np.array([-1, -1, -1]) / np.linalg.norm(np.array([-1, -1, -1]))
    window = Window(
        x_pixel_sz = 50, 
        y_pixel_sz = 12, 
        x_scale = 25.0, 
        y_scale = 10.0,    
        camera_position = np.array([0, 0, -40.0]) ,
        camera_direction = np.array([0, 0, 1.0]),
        camera_x_vector = np.array([1.0, 0, 0]), 
        camera_y_vector = np.array([0, 1.0, 0]),
        focal_length=20.0,
        light_direction=light_direction,
        models={'cube1': cube}, 
    )

    theta = math.pi * 0.05
    gamma = math.pi * 0.02 
    delta = math.pi * 0.04
    rotation_matrix = xy_rot_mat(theta).dot(xz_rot_mat(gamma)).dot(yz_rot_mat(delta))

    for i in range(50):
        time.sleep(0.05)
        cube.transform(rotation_matrix)
        window.refresh_and_render()
        
    for i in range(50):
        time.sleep(0.05)
        light_direction = light_direction.dot(rotation_matrix)   
        window.update_light_direction(light_direction)


    ######################
    # Interactive session
    
    unit_xy = xy_rot_mat(math.pi * 0.05)
    inv_unit_xy = xy_rot_mat(-math.pi * 0.05)

    unit_xz = xz_rot_mat(math.pi * 0.05)
    inv_unit_xz = xz_rot_mat(-math.pi * 0.05)

    unit_yz = yz_rot_mat(math.pi * 0.05)
    inv_unit_yz = yz_rot_mat(-math.pi * 0.05)
    window.render_frame()

    state = {
        'unit_xy': unit_xy,
        'inv_unit_xy': inv_unit_xy,
        'unit_xz': unit_xz,
        'inv_unit_xz': inv_unit_xz,
        'unit_yz': unit_yz,
        'inv_unit_yz': inv_unit_yz,
        'window': window, 
        'cube': cube,
        'light_direction': light_direction,
    }

    async def press(key):
        if key == 'right':
            state['cube'].transform(state['unit_xz'])
            window.refresh_and_render()
        elif key == 'left':    
            state['cube'].transform(state['inv_unit_xz'])
            window.refresh_and_render()

        elif key == 'up':    
            state['cube'].transform(state['inv_unit_yz'])
            window.refresh_and_render()
        elif key == 'down':    
            state['cube'].transform(state['unit_yz'])
            window.refresh_and_render()

        elif key == 'z':    
            state['cube'].transform(state['inv_unit_xy'])
            window.refresh_and_render()
        elif key == 'x':    
            state['cube'].transform(state['unit_xy'])
            window.refresh_and_render()

        ## Light direction update 
        
        if key == 'd':
            state['light_direction'] = state['light_direction'].dot(state['unit_xz'])
            window.update_light_direction(state['light_direction'])
        elif key == 'a':    
            state['light_direction'] = state['light_direction'].dot(state['inv_unit_xz'])
            window.update_light_direction(state['light_direction'])

        elif key == 'w':    
            state['light_direction'] = state['light_direction'].dot(state['inv_unit_yz'])
            window.update_light_direction(state['light_direction'])
        elif key == 's':    
            state['light_direction'] = state['light_direction'].dot(state['unit_yz'])
            window.update_light_direction(state['light_direction'])

        elif key == 'q':    
            state['light_direction'] = state['light_direction'].dot(state['inv_unit_xy'])
            window.update_light_direction(state['light_direction'])
        elif key == 'e':    
            state['light_direction'] = state['light_direction'].dot(state['unit_xy'])
            window.update_light_direction(state['light_direction'])

    listen_keyboard(
        on_press=press,
        delay_second_char = 0.05
    )


if __name__ == "__main__":
    main()