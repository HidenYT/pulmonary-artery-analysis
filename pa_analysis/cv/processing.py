from skan import csr
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import gaussian_filter

from core.config import Config
from pa_analysis.entity import CVResult
import matplotlib.pyplot as plt
import cv2 as cv


def get_normal_unit_vec(points, index, window_sz=5):
    half = window_sz//2
    start = index-half
    end = index+half+1
    if start < 0 or end > len(points):
        return None
    local = points[start:end]
    coords = np.array([[p[1], p[0]] for p in local], dtype=np.float32)
    coords -= np.mean(coords, axis=0)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    tangent = eigvecs[:, 0]
    normal = np.array([tangent[0], tangent[1]])
    return normal / np.linalg.norm(normal)


def cast_until_boundary(mask, origin, direction, max_length=100):
    h, w = mask.shape
    ox, oy = prevx, prevy = x, y = origin
    for i in range(1, max_length):
        prevx = x
        prevy = y
        x = int(round(ox + i * direction[0]))
        y = int(round(oy + i * direction[1]))
        if x < 0 or y < 0 or x >= w or y >= h or mask[y, x] == 0:
            return (prevx, prevy)
    return None


def remove_edge_skeleton_points(skeleton_coordinates, remove_rate):
    remove_1_side = remove_rate/2
    remove_1_side_n = int(len(skeleton_coordinates) * remove_1_side)
    return skeleton_coordinates[remove_1_side_n:-remove_1_side_n]


def find_artery_skeleton_coordinates(mask):
    mask = gaussian_filter(mask, sigma=1) > 0
    skeleton = skeletonize(mask)
    skeleton_graph = csr.Skeleton(skeleton)
    paths = skeleton_graph.paths_list()
    path_lengths = [len(p) for p in paths]
    longest_path = paths[np.argmax(path_lengths)]
    return skeleton_graph.coordinates[longest_path]


def find_width_points_for_artery(artery_segment_coordinates: np.ndarray, normal_vec_window_sz=10, edge_points_remove_ratio=0.5, skeleton_coordinates=None) -> tuple[np.ndarray, np.ndarray]:
    points_centered = artery_segment_coordinates
    inc = points_centered.min(axis=0)
    points_centered -= inc
    mesh = np.zeros((points_centered.max(axis=0)+1), dtype=np.float32)
    for point in points_centered:
        mesh[*point] = 1

    if skeleton_coordinates is None:
        skeleton_coordinates = find_artery_skeleton_coordinates(mesh)
    else:
        skeleton_coordinates = skeleton_coordinates - inc
    skeleton_coordinates = remove_edge_skeleton_points(skeleton_coordinates, remove_rate=edge_points_remove_ratio)
    # plt.scatter(*artery_segment_coordinates.T)
    # plt.scatter(*skeleton_coordinates.T)
    # plt.show()
    max_pair = find_diameter_from_skeleton(mesh, skeleton_coordinates, window_sz=normal_vec_window_sz)
    return max_pair[0] + inc[::-1], max_pair[1] + inc[::-1]


def find_diameter_from_skeleton(mask, skeleton_coords, window_sz=10) -> tuple:
    max_diameter = 0
    max_pair = None

    for i, (y, x) in enumerate(skeleton_coords):
        direction = get_normal_unit_vec(skeleton_coords, i, window_sz=window_sz)
        if direction is None:
            continue
        p1 = cast_until_boundary(mask, (x, y), direction)
        p2 = cast_until_boundary(mask, (x, y), -direction)
        if p1 is None or p2 is None:
            continue
        diameter = np.linalg.norm(np.array(p1) - np.array(p2))
        if diameter > max_diameter:
            max_diameter = diameter
            max_pair = (p1, p2)
    return max_pair


def find_mask_points(mask: Image.Image) -> np.ndarray:
    return np.array(np.where(np.array([mask]) == 1)[1:]).T


def find_arteries(mask_points: np.ndarray, mask_shape: np.ndarray) -> np.ndarray:
    main_artery_init = [0, mask_shape[1]]
    left_artery_init = mask_shape
    right_artery_init = [mask_shape[0], 0]
    kmeans = KMeans(3, init=[main_artery_init, left_artery_init, right_artery_init])
    return kmeans.fit_predict(mask_points)


def find_all_skeleton_paths(mask, top_longest_n=6):
    skeleton = skeletonize(mask)
    skeleton_graph = csr.Skeleton(skeleton)
    paths = skeleton_graph.paths_list()
    sorted_paths = sorted(paths, key=lambda x: len(x), reverse=True)
    take_paths = sorted_paths[:top_longest_n]
    return [skeleton_graph.coordinates[path] for path in take_paths]


def find_best_path_for_clusters(clusters_points, clusters, skeleton_paths_coordinates):
    d = {}
    for c in np.unique(clusters):
        points = clusters_points[clusters==c]
        points_set = {tuple(point) for point in points}
        top_intersection_sz = 0
        top_intersection_idx = None
        for i, path in enumerate(skeleton_paths_coordinates):
            path_points_set = {tuple(point) for point in path}
            intersection = path_points_set & points_set
            if len(intersection) > top_intersection_sz:
                top_intersection_idx = i
                top_intersection_sz = len(intersection)
        d[c] = skeleton_paths_coordinates[top_intersection_idx]
    return d


def find_arteries_d(mask: Image.Image, config: Config) -> CVResult:
    mask_points = find_mask_points(mask)
    clusters = find_arteries(mask_points, mask.shape)

    top_paths = find_all_skeleton_paths(mask)
    best_paths = find_best_path_for_clusters(mask_points, clusters, top_paths)

    cluster_centers = []
    point_pairs = []
    for i in range(3):
        artery_points = mask_points[clusters==i]
        cluster_centers.append(np.mean(artery_points, axis=0))
        width_points = find_width_points_for_artery(
            artery_points, 
            normal_vec_window_sz=config.normal_vector_window_sz,
            edge_points_remove_ratio=config.skeleton_edge_points_remove_ratio,
            skeleton_coordinates=best_paths[i]
        )
        point_pairs.append(width_points)
    cluster_centers = np.array(cluster_centers)

    right_d_idx = np.argmin(cluster_centers[:, 1])
    main_d_idx = np.argmin(cluster_centers[:, 0])
    left_d_idx = list(set(range(3)) - {int(right_d_idx), int(main_d_idx)})[0]
    left_d = point_pairs[left_d_idx]
    main_d = point_pairs[main_d_idx]
    right_d = point_pairs[right_d_idx]
    
    return CVResult(main_artery_points=main_d, left_artery_points=left_d, right_artery_points=right_d)
    