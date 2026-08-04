from skan import csr
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter

from core.config import Config
from pa_analysis.entity import CVResult


def get_normal_unit_vec(points, index, window_sz=5):
    """Поиск нормального вектора в области пикселей в заданном окне"""
    # Выделение пикселей в окне
    half = window_sz//2
    start = index-half
    end = index+half+1
    if start < 0 or end > len(points):
        return None
    local = points[start:end]
    # Метод принципиальных компонент
    coords = np.array([[p[1], p[0]] for p in local], dtype=np.float32)
    coords -= np.mean(coords, axis=0)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    tangent = eigvecs[:, 0]
    normal = np.array([tangent[0], tangent[1]])
    return normal / np.linalg.norm(normal)


def cast_until_boundary(mask, origin, direction, max_length=100):
    """Проведение диаметра методом распространения луча до границы маски"""
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


def remove_center_edge_skeleton_points_from_pivot(skeleton_coordinates, pivot, remove_rate_center, remove_rate_edge):
    """Удаление доли точек скелета относительно двух опорных точек"""
    remove_center_n = int(len(skeleton_coordinates) * remove_rate_center)
    remove_edge_n = int(len(skeleton_coordinates) * remove_rate_edge)
    sorted_coords = sorted(skeleton_coordinates, key=lambda x: np.linalg.norm(x - pivot))
    return np.array(sorted_coords[remove_center_n:-remove_edge_n])


def find_width_points_for_artery(artery_segment_coordinates: np.ndarray, config: Config, skeleton_coordinates, remove_pivot) -> tuple[np.ndarray, np.ndarray]:
    """Поиск крайних точек диаметра артерии"""
    # Смещение в начало координат
    points_centered = artery_segment_coordinates
    inc = points_centered.min(axis=0)
    points_centered -= inc
    mesh = np.zeros((points_centered.max(axis=0)+1), dtype=np.float32)
    for point in points_centered:
        mesh[*point] = 1

    # Смещение скелета в начало координат
    skeleton_coordinates = skeleton_coordinates - inc
    # Удаление точек скелета с края и от центра
    skeleton_coordinates = remove_center_edge_skeleton_points_from_pivot(skeleton_coordinates, remove_pivot-inc, config.skeleton_center_points_remove_ratio, config.skeleton_points_remove_from_edge_ratio)
    # Поиск диаметра
    max_pair = find_diameter_from_skeleton(mesh, skeleton_coordinates, window_sz=config.normal_vector_window_sz)
    return max_pair[0] + inc[::-1], max_pair[1] + inc[::-1]


def find_diameter_from_skeleton(mask, skeleton_coords, window_sz=10) -> tuple:
    """Поиск диаметра артерии вдоль скелета"""
    max_diameter = 0
    max_pair = None

    # Цикл по всем точкам скелета
    for i, (y, x) in enumerate(skeleton_coords):
        # Построение и проведение перпендикуляра к скелету
        direction = get_normal_unit_vec(skeleton_coords, i, window_sz=window_sz)
        if direction is None:
            continue
        p1 = cast_until_boundary(mask, (x, y), direction)
        p2 = cast_until_boundary(mask, (x, y), -direction)
        if p1 is None or p2 is None:
            continue
        # Вычисление длины диаметра
        diameter = np.linalg.norm(np.array(p1) - np.array(p2))
        if diameter > max_diameter:
            max_diameter = diameter
            max_pair = (p1, p2)
    return max_pair


def find_mask_points(mask: Image.Image) -> np.ndarray:
    """Поиск координат пикселей маски"""
    return np.array(np.where(np.array([mask]) == 1)[1:]).T


def find_arteries(mask_points: np.ndarray, mask_shape: np.ndarray) -> np.ndarray:
    """Поиск артерий по общей маске методом К-средних"""
    main_artery_init = [0, mask_shape[1]]
    left_artery_init = mask_shape
    right_artery_init = [mask_shape[0], 0]
    kmeans = KMeans(3, init=[main_artery_init, left_artery_init, right_artery_init])
    return kmeans.fit_predict(mask_points)


def find_all_skeleton_paths(mask, top_longest_n=6):
    """Поиск наиболее длинных отрезков скелета"""
    # Получение всех отрезков
    skeleton = skeletonize(mask)
    skeleton_graph = csr.Skeleton(skeleton)
    paths = skeleton_graph.paths_list()
    # Сортировка и отсечение наиболее длинных
    sorted_paths = sorted(paths, key=lambda x: len(x), reverse=True)
    take_paths = sorted_paths[:top_longest_n]
    return [skeleton_graph.coordinates[path] for path in take_paths]


def find_best_path_for_clusters(clusters_points, clusters, skeleton_paths_coordinates):
    """Выбор наиболее подходящих отрезков скелета для кластеров артерий"""
    d = {}
    # Цикл по кластеру каждой из артерий
    for c in np.unique(clusters):
        # Выбор точек артерии
        points = clusters_points[clusters==c]
        # Создание множества точек для поиска пересечения
        points_set = {tuple(point) for point in points}
        top_intersection_sz = 0
        top_intersection_idx = None
        # Цикл по всем отрезком скелета
        for i, path in enumerate(skeleton_paths_coordinates):
            # Создание множества точек для поиска пересечения
            path_points_set = {tuple(point) for point in path}
            # Оценка размера пересечения и выбор наибольшего
            intersection = path_points_set & points_set
            if len(intersection) > top_intersection_sz:
                top_intersection_idx = i
                top_intersection_sz = len(intersection)
        d[c] = skeleton_paths_coordinates[top_intersection_idx]
    return d


def intersect_paths_with_arteries(clusters_points, clusters, cluster_to_skeleton_coords_map):
    """Обрезание точек отрезков скелета по пересечению с артериями"""
    d = {}
    # Цикл по кластеру каждой из артерий
    for c in np.unique(clusters):
        points = clusters_points[clusters==c]
        points_set = {tuple(point) for point in points}
        path_points_set = {tuple(point) for point in cluster_to_skeleton_coords_map[c]}
        # Создания массива из пересечения точек скелета и кластера
        d[c] = np.array(list(points_set & path_points_set))
    return d


def find_arteries_d(mask: Image.Image, config: Config) -> CVResult:
    """Основная функция поиска диаметра по общей маске сегментации"""
    # Поиск координат масок
    mask_points = find_mask_points(mask)
    # Поиск кластеров артерий
    clusters = find_arteries(mask_points, mask.shape)

    # Поиск наиболее подходящих скелетов для каждой артерии
    top_paths = find_all_skeleton_paths(mask)
    best_paths = find_best_path_for_clusters(mask_points, clusters, top_paths)
    best_paths = intersect_paths_with_arteries(mask_points, clusters, best_paths)

    cluster_centers = []
    point_pairs = []

    # Центр массы маски
    remove_pivot = mask_points.mean(axis=0)
    for i in range(3):
        # Поиск диаметра артерии
        artery_points = mask_points[clusters==i]
        cluster_centers.append(np.mean(artery_points, axis=0))
        width_points = find_width_points_for_artery(
            artery_points, 
            config=config,
            skeleton_coordinates=best_paths[i],
            remove_pivot=remove_pivot,
        )
        point_pairs.append(width_points)
    # Определение того, какая артерия какие координаты имеет
    cluster_centers = np.array(cluster_centers)

    right_d_idx = np.argmin(cluster_centers[:, 1])
    main_d_idx = np.argmin(cluster_centers[:, 0])
    left_d_idx = list(set(range(3)) - {int(right_d_idx), int(main_d_idx)})[0]
    left_d = point_pairs[left_d_idx]
    main_d = point_pairs[main_d_idx]
    right_d = point_pairs[right_d_idx]
    
    return CVResult(main_artery_points=main_d, left_artery_points=left_d, right_artery_points=right_d)
    