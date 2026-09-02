import torch

COCO_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16)
]


def build_spatial_edges(num_nodes=17):
    edges = []

    for src, dst in COCO_EDGES:
        edges.append((src, dst))
        edges.append((dst, src))

    return torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()


def build_temporal_edges(sequence_length=16, num_nodes=17):
    edges = []

    for t in range(sequence_length - 1):
        for node in range(num_nodes):
            current = t * num_nodes + node
            next_node = (t + 1) * num_nodes + node

            edges.append((current, next_node))
            edges.append((next_node, current))

    return torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()


def build_graph(keypoints):
    T, num_nodes, features = keypoints.shape

    node_features = keypoints.reshape(
        T * num_nodes,
        features
    )

    spatial_edges = []

    base_edges = build_spatial_edges(num_nodes)

    for t in range(T):
        offset = t * num_nodes
        frame_edges = base_edges + offset
        spatial_edges.append(frame_edges)

    spatial_edges = torch.cat(
        spatial_edges,
        dim=1
    )

    temporal_edges = build_temporal_edges(
        T,
        num_nodes
    )

    edge_index = torch.cat(
        [
            spatial_edges,
            temporal_edges
        ],
        dim=1
    )

    return node_features, edge_index