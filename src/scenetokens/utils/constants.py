from enum import Enum


MILLION = 1e6
EPSILON = 1e-10
LARGE_FLOAT = 1e10
POSITION_DIMS = [2, 3]
MIN_VALID_POINTS = 2
INVALID_AGENT_ID = -1

DEFAULT_COLLISION_THRESHOLDS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)

# NOTE: borrowed from UniTraj: https://arxiv.org/pdf/2403.15098. They don't explain why these ranges were selected.
KALMAN_DIFFICULTY = {"easy": [0, 30], "medium": [30, 60], "hard": [60, 9999999]}


class ModelStatus(Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


class SampleSelection(Enum):
    ALL = "all"
    RANDOM_DROP = "random_drop"
    TOKEN_RANDOM_DROP = "token_random_drop"  # noqa: S105
    SIMPLE_TOKEN_JACCARD_DROP = "simple_token_jaccard_drop"  # noqa: S105
    SIMPLE_TOKEN_HAMMING_DROP = "simple_token_hamming_drop"  # noqa: S105
    GUMBEL_TOKEN_JACCARD_DROP = "gumbel_token_jaccard_drop"  # noqa: S105
    GUMBEL_TOKEN_HAMMING_DROP = "gumbel_token_hamming_drop"  # noqa: S105
    KMEANS_RANDOM_DROP = "kmeans_random_drop"
    SIMPLE_KMEANS_COSINE_DROP = "simple_kmeans_cosine_drop"
    GUMBEL_KMEANS_COSINE_DROP = "gumbel_kmeans_cosine_drop"
    DEN_TP = "den_tp"


class DataSplits(Enum):
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2


class TrajectoryType(Enum):
    STATIONARY = 0
    STRAIGHT = 1
    STRAIGHT_RIGHT = 2
    STRAIGHT_LEFT = 3
    RIGHT_U_TURN = 4
    RIGHT_TURN = 5
    LEFT_U_TURN = 6
    LEFT_TURN = 7


class AgentBehaviorType(Enum):
    NON_CAUSAL = 0
    CAUSAL = 1


class CausalOutputType(Enum):
    GROUND_TRUTH = 0
    PREDICTION = 1
