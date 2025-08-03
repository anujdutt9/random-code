import logging

import torch
from torch_kmeans import KMeans

logger = logging.getLogger(__name__)


def cluster_kv_vectors(
    activations,
    keys,
    values,
    layer_id,
    n_clusters,
    seed,
    method="kmeans",
    cluster_on="activations",
):
    """
    Cluster the vectors using KMeans.

    Args:
        activations: The activations of all layers
        keys: The keys of all layers
        values: The values of all layers
        layer_id: The layer to cluster on
        n_clusters: The number of clusters
        seed: The seed to use for the KMeans model
        method: The clustering method to use
        cluster_on: The entity to cluster on. Can be "activations", "keys", "values", or "keys+values"

    Returns:
        The centers of the clusters
    """

    # Select the vectors to cluster on and the layer to cluser on
    batch_size = activations[layer_id].size(0)
    if cluster_on == "activations":
        vectors = activations[layer_id]                 # [batch_size, dim]
    elif cluster_on == "keys":
        vectors = keys[layer_id]                        # [batch_size, n_heads, head_dim]
        vectors = vectors.view(batch_size, -1)          # [batch_size, dim]
    elif cluster_on == "values":
        vectors = values[layer_id]                      # [batch_size, n_heads, head_dim]
        vectors = vectors.view(batch_size, -1)          # [batch_size, dim]
    elif cluster_on == "keys+values":
        vectors = torch.cat(
            [
                keys[layer_id].view(batch_size, -1),
                values[layer_id].view(batch_size, -1),
            ],
            dim=1,
        )                                               # [batch_size, 2 * dim]
    else:
        raise ValueError(f"Invalid clustering entity: {cluster_on}")

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, seed=seed)
        result = model(vectors.unsqueeze(0))
    else:
        raise ValueError(f"Invalid clustering method: {method}")

    # Get the centers of the clusters for each layer
    cluster_labels = result.labels.squeeze()
    logger.info(f"Cluster labels: {cluster_labels}")
    keys_centers = []
    values_centers = []

    for i in range(n_clusters):
        keys_dict = {}
        values_dict = {}
        for layer_id in keys:
            keys_dict[layer_id] = keys[layer_id][cluster_labels == i].mean(dim=0)
            values_dict[layer_id] = values[layer_id][cluster_labels == i].mean(dim=0)

        keys_centers.append(keys_dict)
        values_centers.append(values_dict)

    return keys_centers, values_centers, cluster_labels
