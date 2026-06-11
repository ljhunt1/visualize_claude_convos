/**
 * Wire-format contract with the Python side.
 *
 * These types must match the JSON written by `pipeline/export.py`. The
 * inner types (Conversation, Landmark, Tag, ClusterInfo) double as the
 * prop vocabulary for components.
 */

export interface Tag {
  name: string;
  score: number;
}

export interface Conversation {
  /** claude.ai conversation uuid. */
  id: string;
  /** Link to the original conversation on claude.ai. */
  url: string;
  title: string;
  /** ISO date (YYYY-MM-DD). */
  date: string;
  summary: string;
  x: number;
  y: number;
  /** Cluster id, or null for HDBSCAN noise points. */
  cluster: number | null;
  tags: Tag[];
  nChars: number;
  nWords: number;
  nTurns: number;
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
  /** Number of conversations containing at least one member tag. */
  memberCount: number;
}

export interface ClusterInfo {
  id: number;
  /** Short human-readable name (Haiku-generated). */
  name: string;
  size: number;
}

export interface UmapMeta {
  nNeighbors: number;
  minDist: number;
}

export interface DataMeta {
  /** ISO timestamp of when the export was generated. */
  generatedAt: string;
  labelModel: string;
  labelPromptFp: string;
  embedModel: string;
  umap: UmapMeta;
  clusterAlgo: string;
}

export interface UIData {
  conversations: Conversation[];
  landmarks: Landmark[];
  clusters: ClusterInfo[];
  meta: DataMeta;
}
