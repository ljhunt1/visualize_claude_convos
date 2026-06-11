import Plotly from 'plotly.js-basic-dist-min';
import type {
  Config,
  Data,
  Layout,
  PlotMouseEvent,
  PlotRelayoutEvent,
} from 'plotly.js';
import { useEffect, useRef } from 'react';

interface PlotlyChartProps {
  data: Data[];
  layout: Partial<Layout>;
  config: Partial<Config>;
  onPointClick?: (event: PlotMouseEvent) => void;
  onRelayout?: (event: PlotRelayoutEvent) => void;
  className?: string;
}

/**
 * Thin React wrapper around Plotly.react. We roll our own instead of using
 * react-plotly.js, which is a class component pinned to React 16-18 peers.
 */
export function PlotlyChart({
  data,
  layout,
  config,
  onPointClick,
  onRelayout,
  className,
}: PlotlyChartProps) {
  const divRef = useRef<HTMLDivElement>(null);
  const onPointClickRef = useRef(onPointClick);
  const onRelayoutRef = useRef(onRelayout);

  useEffect(() => {
    onPointClickRef.current = onPointClick;
    onRelayoutRef.current = onRelayout;
  });

  useEffect(() => {
    const div = divRef.current;
    if (!div) return;
    void Plotly.react(div, data, layout, config).then((gd) => {
      gd.removeAllListeners('plotly_click');
      gd.removeAllListeners('plotly_relayout');
      gd.on('plotly_click', (event) => {
        onPointClickRef.current?.(event);
      });
      gd.on('plotly_relayout', (event) => {
        onRelayoutRef.current?.(event);
      });
    });
  }, [data, layout, config]);

  useEffect(() => {
    const div = divRef.current;
    return () => {
      if (div) Plotly.purge(div);
    };
  }, []);

  return <div ref={divRef} className={className} />;
}
