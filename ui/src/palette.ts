/**
 * Cluster colors: muted inks that sit well on the paper background.
 * Index 0 is the project's forest green; the rest rotate through
 * survey-map hues. Noise (cluster === null) gets a warm gray.
 */
const CLUSTER_INKS = [
  '#1d6b3c', // forest
  '#a4682a', // ochre
  '#44618c', // slate
  '#8c4a6e', // plum
  '#2a7d78', // teal
  '#a04b38', // rust
  '#5d6a2e', // olive
  '#585a9c', // indigo
] as const;

export const NOISE_INK = '#a39b8b';

export function clusterColor(cluster: number | null): string {
  if (cluster === null) return NOISE_INK;
  return CLUSTER_INKS[cluster % CLUSTER_INKS.length] ?? NOISE_INK;
}

export function clusterLabel(cluster: number | null): string {
  return cluster === null ? 'unclustered' : `cluster ${String(cluster)}`;
}
