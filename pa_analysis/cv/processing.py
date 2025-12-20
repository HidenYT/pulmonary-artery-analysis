from skan import csr
from PIL import Image
from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np

from skimage.morphology import skeletonize

from pa_analysis.entity import CVResult


def get_normal_unit_vec(points, index, window_sz=5):
    half = window_sz//2
    start = max(index-half, 0)
    end = max(index+half+1, len(points))
    local = points[start:end]
    if len(local) < 2:
        return None
    coords = np.array([[p[1], p[0]] for p in local])
    coords -= np.mean(coords, axis=0)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    tangent = eigvecs[:, 0]
    normal = np.array([tangent[0], tangent[1]])
    return normal / np.linalg.norm(normal)


def cast_until_boundary(mask, origin, direction, max_length=100):
    h, w = mask.shape
    ox, oy = prevx, prevy = origin
    x = None
    y = None
    for i in range(1, max_length):
        prevx = x
        prevy = y
        x = int(round(ox + i * direction[0]))
        y = int(round(oy + i * direction[1]))
        if x < 0 or y < 0 or x >= w or y >= h or mask[y, x] == 0:
            return (prevx, prevy)
    return None


def find_width_points_for_artery(artery_segment_coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = artery_segment_coordinates
    inc = center.min(axis=0)
    center -= inc
    mesh = np.zeros((center[:, 0].max()+1, center[:, 1].max()+1))
    for point in center:
        mesh[*point] = 255
    image_8bit = np.uint8(mesh)

    skeleton = skeletonize(image_8bit // 255)
    skeleton_graph = csr.Skeleton(skeleton)
    paths = skeleton_graph.paths_list()
    path_lengths = [len(p) for p in paths]
    longest_path = paths[np.argmax(path_lengths)]
    clean_skeleton = np.zeros_like(skeleton, dtype=bool)
    for y, x in skeleton_graph.coordinates[longest_path]:
        clean_skeleton[y, x] = True

    skeleton_coords = np.argwhere(clean_skeleton).astype(float)
    
    max_diameter = 0
    max_pair = None

    for i, (y, x) in enumerate(skeleton_coords):
        direction = get_normal_unit_vec(skeleton_coords, i, window_sz=10)
        if direction is None:
            continue
        p1 = cast_until_boundary(image_8bit, (x, y), direction)
        p2 = cast_until_boundary(image_8bit, (x, y), -direction)
        if p1 is None or None in p1 or p2 is None or None in p2:
            continue
        diameter = np.linalg.norm(np.array(p1) - np.array(p2))
        if diameter > max_diameter:
            max_diameter = diameter
            max_pair = (p1, p2)
    return max_pair[0] + inc[::-1], max_pair[1] + inc[::-1]


def find_mask_points(mask: Image.Image) -> np.ndarray:
    return np.array(np.where(np.array([mask]) == 255)[1:]).T


def find_arteries(mask_points: np.ndarray) -> np.ndarray:
    kmeans = KMeans(3, n_init=1)
    return kmeans.fit_predict(mask_points)


def find_arteries_d(mask: Image.Image) -> CVResult:
    mask_points = find_mask_points(mask)
    clusters = find_arteries(mask_points)

    cluster_centers = []
    for i in range(3):
        c = mask_points[clusters==i]
        cluster_centers.append(np.mean(c, axis=0))
    cluster_centers = np.array(cluster_centers)

    point_pairs = []
    for i in range(3):
        point_pairs.append(find_width_points_for_artery(mask_points[clusters==i]))

    left_d_idx = np.argmin(cluster_centers[:, 1])
    main_d_idx = np.argmin(cluster_centers[:, 0])
    left_d = point_pairs[left_d_idx]
    main_d = point_pairs[main_d_idx]
    for i in range(3):
        if i not in (left_d_idx, main_d_idx):
            right_d = point_pairs[i]
            break
    return CVResult(main_artery_points=main_d, left_artery_points=left_d, right_artery_points=right_d)
    