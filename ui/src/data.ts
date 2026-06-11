import { useEffect, useState } from 'react';
import type { UIData } from 'types';

/**
 * Light shape check on the fetched JSON — enough to fail loudly if the
 * Python dump and ui/src/types.ts drift apart, not a full validator.
 */
function assertUIData(value: unknown): asserts value is UIData {
  if (typeof value !== 'object' || value === null) {
    throw new Error('data.json: expected a JSON object');
  }
  const obj = value as Record<string, unknown>;
  for (const key of ['conversations', 'landmarks'] as const) {
    if (!Array.isArray(obj[key])) {
      throw new Error(`data.json: missing array field "${key}"`);
    }
  }
  if (typeof obj.meta !== 'object' || obj.meta === null) {
    throw new Error('data.json: missing "meta" object');
  }
}

export async function fetchUIData(): Promise<UIData> {
  const res = await fetch('/data.json');
  if (!res.ok) {
    throw new Error(
      `Failed to fetch /data.json (HTTP ${String(res.status)}). ` +
        'Generate it with: uv run python experiments/dump_test_data_for_ui.py',
    );
  }
  const json: unknown = await res.json();
  assertUIData(json);
  return json;
}

export type DataState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: UIData };

export function useUIData(): DataState {
  const [state, setState] = useState<DataState>({ status: 'loading' });

  useEffect(() => {
    let cancelled = false;
    fetchUIData()
      .then((data) => {
        if (!cancelled) setState({ status: 'ready', data });
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setState({
            status: 'error',
            message: err instanceof Error ? err.message : String(err),
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
