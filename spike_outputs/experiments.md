# Experiments

Hand-curated log of clustering experiments. Each row points at the corresponding output `.md` (if saved).

Output filename convention: `NNN_<path>_<algo>_<centering>.md` — e.g. `004_tags_hdbscan_meancentered.md`.

001. Embed summaries (BGE-base). No mean centering. Cluster HBDSCAN min_cluster=3, cosine distance (5/20/26 2:34pm): [23, 3] + 24 noise (bad). Output: not saved (overwritten before saving was set up)
002. Embed tags (BGE-base, weighted mean conv vector). No mean centering. Cluster HBDSCAN min_cluster=3, cosine distance (5/20/26 2:49pm): [25, 3, 3] + 19 noise (bad). Output: not saved (overwritten before saving was set up)
003. Embed summaries (BGE-base). Mean centering + renormalize. Cluster HBDSCAN min_cluster=3, cosine distance (5/20/26 2:52pm): [25, 12] + 13 noise (mediocre — one mega-cluster). Output: 003_summaries_hdbscan_meancentered.md
004. Embed tags (BGE-base, weighted mean conv vector). Mean centering + renormalize. Cluster HBDSCAN min_cluster=3, cosine distance (5/20/26 2:52pm): [13, 11, 8, 3] + 15 noise (best so far — multiple real clusters, sizes vary). Output: 004_tags_hdbscan_meancentered.md
005. Embed summaries (BGE-base). No mean centering. Cluster Agglomerative n_clusters=8, average linkage, cosine distance (5/20/26 3:20pm): [39, 2, 2, 2, 2, 1, 1, 1] (bad — one mega-cluster, rest singletons/pairs). Output: 005_summaries_agglomerative_nomean.md
006. Embed tags (BGE-base, weighted mean conv vector). No mean centering. Cluster Agglomerative n_clusters=8, average linkage, cosine distance (5/20/26 3:20pm): [26, 8, 5, 5, 3, 1, 1, 1] (bad — mega-cluster but better tail). Output: 006_tags_agglomerative_nomean.md
007. Embed summaries (BGE-base). Mean centering + renormalize. Cluster Agglomerative n_clusters=8, average linkage, cosine distance (5/20/26 3:21pm): [12, 12, 8, 7, 6, 3, 1, 1] (good — well-balanced sizes, no mega-cluster). Output: 007_summaries_agglomerative_meancentered.md
008. Embed tags (BGE-base, weighted mean conv vector). Mean centering + renormalize. Cluster Agglomerative n_clusters=8, average linkage, cosine distance (5/20/26 3:21pm): [14, 10, 9, 7, 4, 3, 2, 1] (good — nice size decay). Output: 008_tags_agglomerative_meancentered.md
