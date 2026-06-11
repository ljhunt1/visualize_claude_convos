/**
 * Wire-format contract with the Python side.
 *
 * These types must match the JSON written by
 * `experiments/dump_test_data_for_ui.py` (currently a one-off test dump;
 * eventually the real pipeline). The inner types (Conversation, Landmark,
 * Tag) double as the prop vocabulary for components.
 */

export interface Tag {
  name: string;
  score: number;
}

export interface Conversation {
  id: string;
  title: string;
  /** ISO date (YYYY-MM-DD). */
  date: string;
  summary: string;
  x: number;
  y: number;
  /** Cluster index, or null for HDBSCAN noise points. */
  cluster: number | null;
  tags: Tag[];
  nChars: number;
  transcript: string;
}

/**
 * A reference tag placed into the same UMAP space as the conversations.
 * Not a canonical feature set — just deduped-ish anchors for orientation.
 */
export interface Landmark {
  name: string;
  x: number;
  y: number;
  /** Number of conversations this tag appeared in. */
  memberCount: number;
}

export interface UmapMeta {
  nNeighbors: number;
  minDist: number;
  meanCenter: boolean;
}

export interface DataMeta {
  /** ISO timestamp of when the dump was generated. */
  generatedAt: string;
  umap: UmapMeta;
  /** Repo-relative path of the clustering experiment the labels came from. */
  clusterSource: string;
}

export interface UIData {
  conversations: Conversation[];
  landmarks: Landmark[];
  meta: DataMeta;
}
