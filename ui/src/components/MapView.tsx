import { PlotlyChart } from 'components/PlotlyChart';
import { clusterColor, clusterLabel } from 'palette';
import type { Config, Data, Layout, PlotData, PlotMouseEvent } from 'plotly.js';
import { useCallback, useMemo } from 'react';
import type { Conversation, Landmark } from 'types';

const MONO = "'IBM Plex Mono', monospace";

interface MapViewProps {
  conversations: Conversation[];
  landmarks: Landmark[];
  /** Conversation ids matching the current search; null = no search active. */
  matchedIds: ReadonlySet<string> | null;
  hiddenClusters: ReadonlySet<number | null>;
  showLandmarks: boolean;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

const LAYOUT: Partial<Layout> = {
  margin: { l: 0, r: 0, t: 0, b: 0 },
  xaxis: { visible: false },
  yaxis: { visible: false },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  showlegend: false,
  dragmode: 'pan',
  // Preserve the user's zoom/pan across data updates (dimming, selection).
  uirevision: 'keep',
  hoverlabel: {
    bgcolor: '#fcfaf3',
    bordercolor: '#d9d2bf',
    font: { family: MONO, size: 12, color: '#20281f' },
  },
};

const CONFIG: Partial<Config> = {
  displayModeBar: false,
  scrollZoom: true,
  responsive: true,
  doubleClick: 'reset',
};

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

export function MapView({
  conversations,
  landmarks,
  matchedIds,
  hiddenClusters,
  showLandmarks,
  selectedId,
  onSelect,
}: MapViewProps) {
  const data = useMemo<Data[]>(() => {
    const traces: Partial<PlotData>[] = [];

    if (showLandmarks) {
      traces.push({
        type: 'scatter',
        mode: 'text',
        x: landmarks.map((lm) => lm.x),
        y: landmarks.map((lm) => lm.y),
        text: landmarks.map((lm) => lm.name),
        textfont: { family: MONO, size: 11, color: 'rgba(78, 86, 72, 0.62)' },
        hoverinfo: 'text',
        hovertext: landmarks.map(
          (lm) => `${lm.name} · ${String(lm.memberCount)} convs`,
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
          color: clusterColor(cluster),
          opacity: convs.map((c) =>
            matchedIds === null || matchedIds.has(c.id) ? 0.88 : 0.12,
          ),
          line: { width: 1, color: '#f6f2e9' },
        },
        hoverinfo: 'text',
        hovertext: convs.map(
          (c) => `${c.title}<br>${c.date} · ${clusterLabel(cluster)}`,
        ),
      });
    }

    const selected = conversations.find((c) => c.id === selectedId);
    if (selected) {
      traces.push({
        type: 'scatter',
        mode: 'markers',
        x: [selected.x],
        y: [selected.y],
        marker: {
          size: 20,
          color: 'rgba(0,0,0,0)',
          line: { width: 2, color: '#20281f' },
        },
        hoverinfo: 'skip',
      });
    }

    return traces;
  }, [
    conversations,
    landmarks,
    matchedIds,
    hiddenClusters,
    showLandmarks,
    selectedId,
  ]);

  const handlePointClick = useCallback(
    (event: PlotMouseEvent) => {
      const id = event.points[0]?.customdata;
      if (typeof id === 'string') onSelect(id);
    },
    [onSelect],
  );

  return (
    <PlotlyChart
      data={data}
      layout={LAYOUT}
      config={CONFIG}
      onPointClick={handlePointClick}
      className="map-plot"
    />
  );
}
