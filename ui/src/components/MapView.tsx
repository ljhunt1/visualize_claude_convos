import { PlotlyChart } from 'components/PlotlyChart';
import type {
  Config,
  Data,
  Layout,
  PlotData,
  PlotMouseEvent,
  PlotRelayoutEvent,
} from 'plotly.js';
import { useCallback, useMemo, useState } from 'react';
import { clusterColor, clusterLabel, type Theme } from 'themes';
import type { Conversation, Landmark } from 'types';

interface Viewport {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

interface MapViewProps {
  theme: Theme;
  conversations: Conversation[];
  landmarks: Landmark[];
  clusterNames: ReadonlyMap<number, string>;
  /** Conversation ids matching the current search; null = no search active. */
  matchedIds: ReadonlySet<string> | null;
  hiddenClusters: ReadonlySet<number | null>;
  showLandmarks: boolean;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

const CONFIG: Partial<Config> = {
  displayModeBar: false,
  scrollZoom: true,
  responsive: true,
  doubleClick: 'reset',
};

/** Most landmark labels shown at once, even fully zoomed in. */
const MAX_LABELS = 28;

function groupByCluster(
  conversations: Conversation[],
): Map<number | null, Conversation[]> {
  const groups = new Map<number | null, Conversation[]>();
  for (const conv of conversations) {
    const group = groups.get(conv.cluster);
    if (group) {
      group.push(conv);
    } else {
      groups.set(conv.cluster, [conv]);
    }
  }
  return groups;
}

/**
 * Map-style label decluttering: prefer high-member-count landmarks, and
 * drop any label that would sit on top of an already-placed one at the
 * current zoom level. Zooming in widens the data-space between labels,
 * so more of them appear — like place names on a slippy map.
 */
function cullLandmarks(landmarks: Landmark[], viewport: Viewport): Landmark[] {
  const vw = viewport.x1 - viewport.x0;
  const vh = viewport.y1 - viewport.y0;
  const inView = landmarks.filter(
    (lm) =>
      lm.x >= viewport.x0 &&
      lm.x <= viewport.x1 &&
      lm.y >= viewport.y0 &&
      lm.y <= viewport.y1,
  );
  const byCount = [...inView].sort((a, b) => b.memberCount - a.memberCount);

  const placed: Landmark[] = [];
  for (const lm of byCount) {
    if (placed.length >= MAX_LABELS) break;
    const collides = placed.some((other) => {
      // Approximate label extents from name lengths (in viewport fractions).
      const minDx = ((lm.name.length + other.name.length) / 2) * 0.011 * vw;
      const minDy = 0.05 * vh;
      return (
        Math.abs(lm.x - other.x) < minDx && Math.abs(lm.y - other.y) < minDy
      );
    });
    if (!collides) placed.push(lm);
  }
  return placed;
}

function fullExtent(
  conversations: Conversation[],
  landmarks: Landmark[],
): Viewport {
  const xs = [...conversations.map((c) => c.x), ...landmarks.map((lm) => lm.x)];
  const ys = [...conversations.map((c) => c.y), ...landmarks.map((lm) => lm.y)];
  return {
    x0: Math.min(...xs),
    x1: Math.max(...xs),
    y0: Math.min(...ys),
    y1: Math.max(...ys),
  };
}

export function MapView({
  theme,
  conversations,
  landmarks,
  clusterNames,
  matchedIds,
  hiddenClusters,
  showLandmarks,
  selectedId,
  onSelect,
}: MapViewProps) {
  const [viewport, setViewport] = useState<Viewport | null>(null);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      margin: { l: 0, r: 0, t: 0, b: 0 },
      xaxis: { visible: false },
      yaxis: { visible: false },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      showlegend: false,
      dragmode: 'pan',
      // Preserve the user's zoom/pan across data updates (dimming, culling).
      uirevision: 'keep',
      hoverlabel: {
        bgcolor: theme.plot.hoverBg,
        bordercolor: theme.plot.hoverBorder,
        font: { family: theme.plot.hoverFont, size: 12.5 },
      },
    }),
    [theme],
  );

  const data = useMemo<Data[]>(() => {
    const traces: Partial<PlotData>[] = [];

    if (showLandmarks) {
      const visible = cullLandmarks(
        landmarks,
        viewport ?? fullExtent(conversations, landmarks),
      );
      traces.push({
        type: 'scatter',
        mode: 'text+markers',
        x: visible.map((lm) => lm.x),
        y: visible.map((lm) => lm.y),
        text: visible.map((lm) => lm.name),
        textposition: 'top center',
        textfont: {
          family: theme.plot.landmarkFont,
          size: 11,
          color: theme.plot.landmarkColor,
        },
        marker: {
          symbol: 'cross-thin',
          size: 7,
          color: 'rgba(0,0,0,0)',
          line: { width: 1.2, color: theme.plot.landmarkColor },
        },
        hoverinfo: 'text',
        hovertext: visible.map(
          (lm) =>
            `landmark tag: ${lm.name}<br>in ${String(lm.memberCount)} conversations`,
        ),
      });
    }

    for (const [cluster, convs] of groupByCluster(conversations)) {
      if (hiddenClusters.has(cluster)) continue;
      traces.push({
        type: 'scatter',
        mode: 'markers',
        x: convs.map((c) => c.x),
        y: convs.map((c) => c.y),
        customdata: convs.map((c) => c.id),
        marker: {
          size: 10,
          color: clusterColor(theme, cluster),
          opacity: convs.map((c) =>
            matchedIds === null || matchedIds.has(c.id) ? 0.9 : 0.12,
          ),
          line: { width: 1, color: theme.plot.markerOutline },
        },
        hoverinfo: 'text',
        hovertext: convs.map(
          (c) =>
            `${c.title}<br>${c.date} · ${clusterLabel(cluster, clusterNames)}`,
        ),
      });
    }

    const selected = conversations.find((c) => c.id === selectedId);
    if (selected) {
      // Plotly markers are pixel-sized; scale the ring with zoom so it
      // hugs the dot at full extent and grows as you zoom in.
      const extent = fullExtent(conversations, landmarks);
      const zoom = viewport
        ? (extent.x1 - extent.x0) / (viewport.x1 - viewport.x0)
        : 1;
      traces.push({
        type: 'scatter',
        mode: 'markers',
        x: [selected.x],
        y: [selected.y],
        marker: {
          size: Math.min(30, 14 * Math.max(1, zoom)),
          color: 'rgba(0,0,0,0)',
          line: { width: 2, color: theme.plot.selectionRing },
        },
        hoverinfo: 'skip',
      });
    }

    return traces;
  }, [
    theme,
    conversations,
    landmarks,
    clusterNames,
    matchedIds,
    hiddenClusters,
    showLandmarks,
    selectedId,
    viewport,
  ]);

  const handlePointClick = useCallback(
    (event: PlotMouseEvent) => {
      const id = event.points[0]?.customdata;
      if (typeof id === 'string') onSelect(id);
    },
    [onSelect],
  );

  const handleRelayout = useCallback((event: PlotRelayoutEvent) => {
    const x0 = event['xaxis.range[0]'];
    const x1 = event['xaxis.range[1]'];
    const y0 = event['yaxis.range[0]'];
    const y1 = event['yaxis.range[1]'];
    if (
      typeof x0 === 'number' &&
      typeof x1 === 'number' &&
      typeof y0 === 'number' &&
      typeof y1 === 'number'
    ) {
      setViewport({ x0, x1, y0, y1 });
    } else if (event['xaxis.autorange'] || event['yaxis.autorange']) {
      setViewport(null);
    }
  }, []);

  return (
    <PlotlyChart
      data={data}
      layout={layout}
      config={CONFIG}
      onPointClick={handlePointClick}
      onRelayout={handleRelayout}
      className="map-plot"
    />
  );
}
