import numpy as np


def create_tiling(feat_range, bins, offset):
    """
    Create 1 tiling spec of 1 dimension(feature)
    feat_range: feature range; example: [-1, 1]
    bins: number of bins for that feature; example: 10
    offset: offset for that feature; example: 0.2
    """

    return np.linspace(feat_range[0], feat_range[1], bins + 1)[1:-1] + offset


def create_tilings(feat_ranges, number_tilings, bins, offsets):
    """

    @param feat_ranges: range of each feature, e.g. x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
    @param number_tilings: number of tilings; example: 3 tilings
    @param bins: bin(tile) size of each tiling and dimension, e.g. [[10, 10], [10, 10], [10, 10]]: 3 tilings * [x_bin, y_bin]
    @param offsets: offset for each tiling and dimensions, e.g. [[0, 0], [0.2, 1], [0.4, 1.5]]: 3 tilings * [x_offset, y_offset]
    """
    tilings = []
    for tiling_i in range(number_tilings):
        tiling_bin = bins[tiling_i]
        tiling_offset = offsets[tiling_i]

        tiling = []
        # for each feature dimension
        for feat_i in range(len(feat_ranges)):
            feat_range = feat_ranges[feat_i]
            # tiling for this feature
            feat_tiling = create_tiling(feat_range, tiling_bin[feat_i], tiling_offset[feat_i])
            tiling.append(feat_tiling)
        tilings.append(tiling)
    return np.array(tilings)


def get_tile_coding(feature, tilings):
    """

    @param feature: sample feature with multiple dimensions that need to be encoded; e.g.: [0.1, 2.5], [-0.3, 2.0]
    @param tilings: tilings with a few layers (3x2x9)
    """
    num_dims = len(feature)
    feat_codings = []
    for tiling in tilings:
        feat_coding = []
        for i in range(num_dims):
            feat_i = feature[i]
            tiling_i = tiling[i]
            coding_i = np.digitize(feat_i, tiling_i)
            feat_coding.append(coding_i)
        feat_codings.append(feat_coding)

    return np.array(feat_codings)


class QValueFunction:
    def __init__(self, tilings, actions, lr):
        self.tilings = tilings
        self.num_tilings = len(self.tilings)
        self.actions = actions
        self.lr = lr  # /self.num_tilings  # learning rate equally assigned to each tiling
        self.state_sizes = [tuple(len(splits) + 1 for splits in tiling) for tiling in
                            self.tilings]  # [(10, 10), (10, 10), (10, 10)]
        self.q_tables = [np.zeros(shape=(state_size + (len(self.actions),))) for state_size in self.state_sizes]


feature_ranges = [[-1, 1], [2, 5]]
number_tilings = 3
bins = [[10, 10], [10, 10], [10, 10]]
offsets = [[0, 0], [0.2, 1], [0.4, 1.5]]

tilings = create_tilings(feature_ranges, number_tilings, bins, offsets)

feature = [0.1, 2.5]

coding = get_tile_coding(feature, tilings)  # index of the state in each tiling
print(coding)
