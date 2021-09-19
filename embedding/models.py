from .losses import *
from .base_model import KnowledgeGraphEmbeddingModel, get_initializer


class MSTE(KnowledgeGraphEmbeddingModel):
    def __init__(self, em_size=100, batch_size=128, nb_epochs=100, initialiser="xavier_uniform", nb_negs=2, margin=1.0,
                 optimiser="amsgrad", lr=0.01, similarity="l1", nb_ents=0, nb_rels=0, reg_wt=0.01, loss="default",
                 seed=1234, verbose=1, log_interval=5):
        """ Initialise new instance of the TriModel model

        Parameters
        ----------
        em_size: int
            embedding vector size
        batch_size: int
            batch size
        nb_epochs: int
            number of epoch i.e training iterations
        initialiser: str
            initialiser name e.g. xavier_uniform or he_normal
        nb_negs: int
            number of negative instance per each positive training instance
        optimiser: str
            optimiser name
        lr: float
            optimiser learning rate
        nb_ents: int
            total number of knowledge graph entities
        nb_rels: int
            total number of knowledge graph relations
        reg_wt: float
            regularisation parameter weight
        seed: int
            random seed
        verbose: int
            verbosity level. options are {0, 1, 2}
        log_interval: int
            the number of epochs to wait until reporting the next training loss. (loss logging frequency)
        """
        super().__init__(em_size=em_size, batch_size=batch_size, nb_epochs=nb_epochs, initialiser=initialiser,
                         nb_negs=nb_negs, optimiser=optimiser, lr=lr, loss=loss, nb_ents=nb_ents, nb_rels=nb_rels,
                         reg_wt=reg_wt, seed=seed, verbose=verbose, log_interval=log_interval)
        self.margin = margin
        self.similarity = similarity


    def score_triples(self, sub_em, rel_em, obj_em):
        """ Compute TriModel scores for a set of triples given their component _embeddings

        Parameters
        ----------
        sub_em: tf.tensor
            Embeddings of the subject entities
        rel_em: tf.tensor
            Embeddings of the relations
        obj_em: tf.tensor
            Embeddings of the object entities

        Returns
        -------
        tf.tensor
            model scores for the original triples of the given _embeddings
        """

        sub_w = tf.sin(rel_em * obj_em)
        rel_w = tf.sin(sub_em * obj_em)
        obj_w = tf.sin(sub_em * rel_em)


        # compute the interaction vectors of the tri-vector _embeddings
        em_interactions = sub_em*sub_w + rel_em*rel_w - obj_em*obj_w
        if self.similarity.lower() == "l1":
            scores = tf.norm(em_interactions, ord=1, axis=1)
        elif self.similarity.lower() == "l2":
            scores = tf.norm(em_interactions, ord=2, axis=1)
        else:
            raise ValueError("Unknown similarity type (%s)." % self.similarity)

        # the use of negative score complies with loss objective
        return -scores



    def compute_loss(self, scores, *args, **kwargs):
        """ Compute TransE training loss using the pairwise hinge loss

        Parameters
        ----------
        scores: tf.Tenor
            scores tensor
        args: list
            Non-Key arguments
        kwargs: dict
            Key arguments

        Returns
        -------
        tf.float32
            model loss value
        """
        if self.loss == "default":
            pos_scores, neg_scores = tf.split(scores, num_or_size_splits=2)
            return pairwise_logistic_loss(pos_scores, neg_scores, reduction_type="avg")
        else:
            return compute_kge_loss(scores, self.loss, reduction_type="avg")



