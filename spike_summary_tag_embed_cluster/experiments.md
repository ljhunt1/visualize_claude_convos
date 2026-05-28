# Experiments

Hand-curated log of clustering experiments. Each row points at the corresponding output directory.

Output directory convention: `NNN_<path>_<algo>_<centering>/` — e.g. `004_tags_hdbscan_meancentered/`. Each directory contains `summary.md`, `conv_vecs.npy`, `labels.npy`, `meta.json` (and `mean_vec.npy` if mean-centered).

1. Embed summaries (BGE-base). No mean centering. Cluster HBDSCAN min_cluster=3, cosine distance (5/20/26 2:34pm): [23, 3] + 24 noise (bad). Output: 001_summaries_hdbscan_nomean/
2. Embed tags (BGE-base, weighted mean conv vector). No mean centering. Cluster HBDSCAN min_cluster=3, cosine distance (5/20/26 2:49pm): [25, 3, 3] + 19 noise (bad). Output: 002_tags_hdbscan_nomean/
3. Embed summaries (BGE-base). Mean centering + renormalize. Cluster HBDSCAN min_cluster=3, cosine distance (5/20/26 2:52pm): [25, 12] + 13 noise (mediocre — one mega-cluster). Output: 003_summaries_hdbscan_meancentered/
4. Embed tags (BGE-base, weighted mean conv vector). Mean centering + renormalize. Cluster HBDSCAN min_cluster=3, cosine distance (5/20/26 2:52pm): [13, 11, 8, 3] + 15 noise (best so far — multiple real clusters, sizes vary). Output: 004_tags_hdbscan_meancentered/
5. Embed summaries (BGE-base). No mean centering. Cluster Agglomerative n_clusters=8, average linkage, cosine distance (5/20/26 3:20pm): [39, 2, 2, 2, 2, 1, 1, 1] (bad — one mega-cluster, rest singletons/pairs). Output: 005_summaries_agglomerative_nomean/
6. Embed tags (BGE-base, weighted mean conv vector). No mean centering. Cluster Agglomerative n_clusters=8, average linkage, cosine distance (5/20/26 3:20pm): [26, 8, 5, 5, 3, 1, 1, 1] (bad — mega-cluster but better tail). Output: 006_tags_agglomerative_nomean/
7. Embed summaries (BGE-base). Mean centering + renormalize. Cluster Agglomerative n_clusters=8, average linkage, cosine distance (5/20/26 3:21pm): [12, 12, 8, 7, 6, 3, 1, 1] (good — well-balanced sizes, no mega-cluster). Output: 007_summaries_agglomerative_meancentered/
8. Embed tags (BGE-base, weighted mean conv vector). Mean centering + renormalize. Cluster Agglomerative n_clusters=8, average linkage, cosine distance (5/20/26 3:21pm): [14, 10, 9, 7, 4, 3, 2, 1] (good — nice size decay). Output: 008_tags_agglomerative_meancentered/

Subjective read (human reviewer):
3 is trash - two megaclusters
5 is trash - megacluster insanely big
6 is trash - megacluster

4 - very solid. A few reasonable clusters
7 - just ok. Python coding type questions are distributed across many clusters
8 - solid. Two categories for travel stuff, less clean split ML questions vs coding questions vs application questions
