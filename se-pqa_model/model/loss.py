from torch import einsum, nn
from torch.nn.functional import relu

class TripletMarginLoss(nn.Module):
    """
    Triplet Margin Loss function.
    """

    def __init__(self, margin=1.0):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(
        self,
        anchors,
        positives,
        negatives,
    ):
        positive_embedding_scores = einsum("xz,xz->x", anchors, positives)
        negative_embedding_scores = einsum("xz,xz->x", anchors, negatives)

        loss = relu(
            self.margin - positive_embedding_scores + negative_embedding_scores
        ).mean()

        return loss