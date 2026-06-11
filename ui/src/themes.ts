/**
 * Design flavors. Each theme owns the things CSS variables can't reach:
 * the plotly trace colors, the fonts plotly renders text in, and the
 * masthead copy. Everything else lives in per-theme blocks in index.css,
 * keyed off the `data-theme` attribute.
 */

export interface PlotStyle {
  hoverBg: string;
  hoverBorder: string;
  hoverText: string;
  hoverFont: string;
  /** Outline around conversation dots, usually the page background. */
  markerOutline: string;
  selectionRing: string;
  landmarkColor: string;
  landmarkFont: string;
}

export interface Theme {
  id: string;
  /** Shown in the flavor picker. */
  label: string;
  appName: string;
  tagline: string;
  clusterInks: readonly string[];
  noiseInk: string;
  plot: PlotStyle;
}

const MONO = "'IBM Plex Mono', monospace";

export const THEMES: readonly Theme[] = [
  {
    id: 'field-atlas',
    label: 'field atlas',
    appName: 'Conversation Atlas',
    tagline: 'claude.ai history, surveyed',
    clusterInks: [
      '#1d6b3c', // forest
      '#a4682a', // ochre
      '#44618c', // slate
      '#8c4a6e', // plum
      '#2a7d78', // teal
      '#a04b38', // rust
      '#5d6a2e', // olive
      '#585a9c', // indigo
    ],
    noiseInk: '#a39b8b',
    plot: {
      hoverBg: '#fcfaf3',
      hoverBorder: '#d9d2bf',
      hoverText: '#20281f',
      hoverFont: MONO,
      markerOutline: '#f6f2e9',
      selectionRing: '#20281f',
      landmarkColor: 'rgba(78, 86, 72, 0.6)',
      landmarkFont: MONO,
    },
  },
  {
    id: 'terminal',
    label: 'terminal',
    appName: 'convo_map',
    tagline: '// my claude chats, umapped',
    clusterInks: [
      '#4af28a', // phosphor green
      '#ffb000', // amber
      '#39d8ff', // cyan
      '#ff6ad5', // magenta
      '#d4e157', // lime
      '#ff7043', // orange
      '#b39dff', // violet
      '#26c6a5', // teal
    ],
    noiseInk: '#3f5a46',
    plot: {
      hoverBg: '#0d130d',
      hoverBorder: '#2c4631',
      hoverText: '#9fe8b4',
      hoverFont: MONO,
      markerOutline: '#070b07',
      selectionRing: '#e8ffe8',
      landmarkColor: 'rgba(110, 168, 124, 0.55)',
      landmarkFont: MONO,
    },
  },
  {
    id: 'notebook',
    label: 'notebook',
    appName: 'my claude chats, mapped',
    tagline: 'a lil side project',
    clusterInks: [
      '#3b82f6', // ballpoint blue
      '#e8537a', // pink marker
      '#22a04c', // green marker
      '#f59e0b', // orange
      '#8b5cf6', // purple
      '#14b8a6', // teal
      '#ef4444', // red
      '#a16207', // brown
    ],
    noiseInk: '#b8b3ab',
    plot: {
      hoverBg: '#fffef8',
      hoverBorder: '#d8d3c8',
      hoverText: '#3a372f',
      hoverFont: "'Shantell Sans Variable', cursive",
      markerOutline: '#fdfcf7',
      selectionRing: '#3a372f',
      landmarkColor: 'rgba(120, 116, 105, 0.65)',
      landmarkFont: "'Shantell Sans Variable', cursive",
    },
  },
  {
    id: 'swiss',
    label: 'swiss',
    appName: 'chat map.',
    tagline: 'fifty conversations, two dimensions',
    clusterInks: [
      '#1d4ed8', // cobalt
      '#e11d48', // red
      '#f59e0b', // amber
      '#059669', // green
      '#7c3aed', // violet
      '#0891b2', // cyan
      '#db2777', // pink
      '#65a30d', // lime
    ],
    noiseInk: '#c4c4c4',
    plot: {
      hoverBg: '#ffffff',
      hoverBorder: '#111111',
      hoverText: '#111111',
      hoverFont: "'Archivo Variable', sans-serif",
      markerOutline: '#ffffff',
      selectionRing: '#111111',
      landmarkColor: 'rgba(17, 17, 17, 0.45)',
      landmarkFont: "'Archivo Variable', sans-serif",
    },
  },
  {
    id: 'observatory',
    label: 'observatory',
    appName: 'night chart',
    tagline: 'the chats, charted',
    clusterInks: [
      '#ffd166', // star gold
      '#8ecae6', // pale blue
      '#f4978e', // dawn rose
      '#b5e48c', // aurora green
      '#cdb4f6', // lilac
      '#90e0ef', // ice
      '#f9c74f', // amber
      '#f4a3c4', // pink
    ],
    noiseInk: '#43507a',
    plot: {
      hoverBg: '#101631',
      hoverBorder: '#2c3a66',
      hoverText: '#cfd9f4',
      hoverFont: MONO,
      markerOutline: '#0a0f24',
      selectionRing: '#f0f4ff',
      landmarkColor: 'rgba(150, 168, 214, 0.6)',
      landmarkFont: "'Cormorant Variable', serif",
    },
  },
];

export const DEFAULT_THEME: Theme = THEMES[0];

export function themeById(id: string | null): Theme {
  return THEMES.find((theme) => theme.id === id) ?? DEFAULT_THEME;
}

export function clusterColor(theme: Theme, cluster: number | null): string {
  if (cluster === null) return theme.noiseInk;
  return (
    theme.clusterInks[cluster % theme.clusterInks.length] ?? theme.noiseInk
  );
}

export function clusterLabel(cluster: number | null): string {
  return cluster === null ? 'unclustered' : `cluster ${String(cluster)}`;
}
