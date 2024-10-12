from .data_loading import get_classification_dataset, get_graph_ensemble_dataset
from .fuzzy_laplacian import get_fuzzy_laplacian
from .misc import set_seed, use_best_hyperparams, get_edge_index_and_theta
from .losses import masked_regression_loss

from .data_utils.triangular_lattice import (
    create_triangular_lattice,
    is_equilateral_triangle,
    create_equilateral_triangular_lattice,

    aggr,
    compute_fuzzy_laplacian,
    propagate_features_fuzzy,
    generate_features_fuzzy,

    single_source_single_sink,
    solenoidal_vectorfield,
)

from .data_utils.gene_regulatory_network import (
    generate_parameters,
    step,
)

try:
    import jax
    from .jax_util import FuzzyDirGCN as JaxFuzzyDirGCN
except ImportError:
    pass