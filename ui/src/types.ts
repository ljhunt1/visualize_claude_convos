export interface Tag {
  name: string;
  score: number;
}

export interface Conversation {
  filename: string;
  title: string;
  date: string;
  summary: string;
  x: number;
  y: number;
  cluster: number;
  tags: Tag[];
  n_chars: number;
}

export interface Feature {
  name: string;
  x: number;
  y: number;
  member_count: number;
}

export interface UmapMeta {
  n_neighbors: number;
  min_dist: number;
  mean_center: boolean;
}

export interface DataMeta {
  n_conversations: number;
  n_features: number;
  umap: UmapMeta;
  cluster_source: string;
}

export interface UIData {
  conversations: Conversation[];
  features: Feature[];
  meta: DataMeta;
}
