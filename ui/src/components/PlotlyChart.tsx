import Plotly from 'plotly.js-basic-dist-min';
import type { Config, Data, Layout, PlotMouseEvent } from 'plotly.js';
import { useEffect, useRef } from 'react';

interface PlotlyChartProps {
  data: Data[];
  layout: Partial<Layout>;
  config: Partial<Config>;
  onPointClick?: (event: PlotMouseEvent) => void;
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
  className,
}: PlotlyChartProps) {
  const divRef = useRef<HTMLDivElement>(null);
  const onPointClickRef = useRef(onPointClick);

  useEffect(() => {
    onPointClickRef.current = onPointClick;
  });

  useEffect(() => {
    const div = divRef.current;
    if (!div) return;
    void Plotly.react(div, data, layout, config).then((gd) => {
      gd.removeAllListeners('plotly_click');
      gd.on('plotly_click', (event) => {
        onPointClickRef.current?.(event);
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
