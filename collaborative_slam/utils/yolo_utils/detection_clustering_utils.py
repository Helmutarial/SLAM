"""
Utility functions for detection clustering and visualization.
"""
import numpy as np
from sklearn.cluster import DBSCAN

def cluster_detections_by_class(detections, min_confidence=0.5, eps=1.2):
    """
    Groups detections by class and clusters them spatially using DBSCAN.
    Returns a dict: {class_name: [centroid1, centroid2, ...]}
    Each centroid is weighted by confidence.
    """
    clustered = {}
    detections_by_class = {}
    for det in detections:
        conf = det.get('confidence', 0)
        if conf < min_confidence:
            continue
        class_name = det.get('class', 'unknown')
        if 'closest_point' in det and det['closest_point'] is not None:
            x, y, z = det['closest_point']
            if class_name not in detections_by_class:
                detections_by_class[class_name] = {'points': [], 'confidences': []}
            detections_by_class[class_name]['points'].append([x, y])
            detections_by_class[class_name]['confidences'].append(conf)
    for class_name, data in detections_by_class.items():
        points_np = np.array(data['points'])
        confs_np = np.array(data['confidences'])
        if len(points_np) == 0:
            continue
        db = DBSCAN(eps=eps, min_samples=1).fit(points_np)
        labels = db.labels_
        unique_labels = set(labels)
        centroids = []
        for label in unique_labels:
            idxs = np.where(labels == label)[0]
            group_points = points_np[idxs]
            group_conf = confs_np[idxs]
            if group_conf.sum() > 0:
                centroid = np.average(group_points, axis=0, weights=group_conf)
            else:
                centroid = group_points.mean(axis=0)
            centroids.append(centroid)
        clustered[class_name] = centroids
    return clustered
