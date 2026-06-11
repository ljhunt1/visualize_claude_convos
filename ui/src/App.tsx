import { DetailPanel } from 'components/DetailPanel';
import { MapView } from 'components/MapView';
import { useUIData } from 'data';
import { clusterColor, clusterLabel } from 'palette';
import { useEffect, useMemo, useState } from 'react';
import type { Conversation, UIData } from 'types';

function matchesQuery(conv: Conversation, query: string): boolean {
  const haystack = [
    conv.title,
    conv.summary,
    conv.date,
    ...conv.tags.map((t) => t.name),
  ]
    .join('\n')
    .toLowerCase();
  return haystack.includes(query);
}

function clusterIds(conversations: Conversation[]): (number | null)[] {
  const seen = new Set<number | null>();
  for (const conv of conversations) seen.add(conv.cluster);
  return [...seen].sort(
    (a, b) => (a ?? Number.MAX_SAFE_INTEGER) - (b ?? Number.MAX_SAFE_INTEGER),
  );
}

function Atlas({ data }: { data: UIData }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [hiddenClusters, setHiddenClusters] = useState<
    ReadonlySet<number | null>
  >(() => new Set());

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setSelectedId(null);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
  }, []);

  const matchedIds = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return null;
    return new Set(
      data.conversations
        .filter((conv) => matchesQuery(conv, needle))
        .map((conv) => conv.id),
    );
  }, [data.conversations, query]);

  const selected = useMemo(
    () => data.conversations.find((conv) => conv.id === selectedId) ?? null,
    [data.conversations, selectedId],
  );

  const toggleCluster = (cluster: number | null) => {
    setHiddenClusters((prev) => {
      const next = new Set(prev);
      if (next.has(cluster)) {
        next.delete(cluster);
      } else {
        next.add(cluster);
      }
      return next;
    });
  };

  return (
    <div className="app">
      <header className="masthead">
        <h1 className="masthead-title">
          Conversation <em>Atlas</em>
        </h1>
        <p className="masthead-stats">
          {data.conversations.length} conversations · {data.landmarks.length}{' '}
          landmarks
        </p>
      </header>

      <div className="toolbar">
        <div className="search">
          <input
            type="search"
            value={query}
            placeholder="search titles, summaries, tags…"
            onChange={(event) => {
              setQuery(event.target.value);
            }}
          />
          {matchedIds !== null && (
            <span className="search-count">
              {matchedIds.size} / {data.conversations.length}
            </span>
          )}
        </div>

        <ul className="chips">
          {clusterIds(data.conversations).map((cluster) => {
            const hidden = hiddenClusters.has(cluster);
            return (
              <li key={cluster ?? 'noise'}>
                <button
                  type="button"
                  className={hidden ? 'chip chip-off' : 'chip'}
                  onClick={() => {
                    toggleCluster(cluster);
                  }}
                >
                  <span
                    className="cluster-dot"
                    style={{ background: clusterColor(cluster) }}
                  />
                  {clusterLabel(cluster)}
                </button>
              </li>
            );
          })}
        </ul>

        <label className="landmark-toggle">
          <input
            type="checkbox"
            checked={showLandmarks}
            onChange={(event) => {
              setShowLandmarks(event.target.checked);
            }}
          />
          landmarks
        </label>
      </div>

      <main className="main">
        <div className="map">
          <MapView
            conversations={data.conversations}
            landmarks={data.landmarks}
            matchedIds={matchedIds}
            hiddenClusters={hiddenClusters}
            showLandmarks={showLandmarks}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </div>
        <DetailPanel
          data={data}
          selected={selected}
          onClose={() => {
            setSelectedId(null);
          }}
        />
      </main>
    </div>
  );
}

function App() {
  const state = useUIData();

  if (state.status === 'loading') {
    return <div className="screen-message">surveying…</div>;
  }
  if (state.status === 'error') {
    return (
      <div className="screen-message screen-error">
        <p>Could not load the atlas.</p>
        <pre>{state.message}</pre>
      </div>
    );
  }
  return <Atlas data={state.data} />;
}

export default App;
