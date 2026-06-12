import { DetailPanel } from 'components/DetailPanel';
import { MapView } from 'components/MapView';
import { TimeBar } from 'components/TimeBar';
import { useUIData } from 'data';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { clusterColor, clusterLabel, THEMES, themeById } from 'themes';
import type { Conversation, UIData } from 'types';

const THEME_STORAGE_KEY = 'convo-map-theme';

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
  const [theme, setTheme] = useState(() =>
    themeById(localStorage.getItem(THEME_STORAGE_KEY)),
  );
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [dateRange, setDateRange] = useState<{
    start: string;
    end: string;
  } | null>(null);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [hiddenClusters, setHiddenClusters] = useState<
    ReadonlySet<number | null>
  >(() => new Set());

  useEffect(() => {
    document.title = theme.appName;
    localStorage.setItem(THEME_STORAGE_KEY, theme.id);
  }, [theme]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setSelectedId(null);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
  }, []);

  const clusterNames = useMemo(
    () => new Map(data.clusters.map((c) => [c.id, c.name])),
    [data.clusters],
  );

  // Conversations are sorted by date in the export.
  const dateExtent = useMemo(() => {
    const dates = data.conversations.map((c) => c.date);
    return {
      min: dates[0] ?? '2024-01-01',
      max: dates[dates.length - 1] ?? '2026-01-01',
    };
  }, [data.conversations]);

  const handleDateChange = useCallback(
    (start: string, end: string) => {
      setDateRange(
        start <= dateExtent.min && end >= dateExtent.max
          ? null
          : { start, end },
      );
    },
    [dateExtent],
  );

  // Search and date range compose into one dim-the-rest filter set.
  const matchedIds = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle && !dateRange) return null;
    return new Set(
      data.conversations
        .filter(
          (conv) =>
            (!needle || matchesQuery(conv, needle)) &&
            (!dateRange ||
              (conv.date >= dateRange.start && conv.date <= dateRange.end)),
        )
        .map((conv) => conv.id),
    );
  }, [data.conversations, query, dateRange]);

  const selected = useMemo(
    () => data.conversations.find((conv) => conv.id === selectedId) ?? null,
    [data.conversations, selectedId],
  );

  // Clicking the already-selected conversation toggles it off.
  const handleSelect = useCallback((id: string) => {
    setSelectedId((prev) => (prev === id ? null : id));
  }, []);

  const toggleCluster = (cluster: number | null) => {
    // Hiding a cluster also deselects any conversation inside it.
    const hiding = !hiddenClusters.has(cluster);
    if (hiding && selected?.cluster === cluster) {
      setSelectedId(null);
    }
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
    <div className="app" data-theme={theme.id}>
      <header className="masthead">
        <div className="masthead-name">
          <h1 className="masthead-title">{theme.appName}</h1>
          <p className="masthead-tagline">{theme.tagline}</p>
        </div>
        <div className="masthead-side">
          <p className="masthead-stats">
            {data.conversations.length} conversations · {data.landmarks.length}{' '}
            landmarks
          </p>
          <label className="flavor-picker">
            flavor
            <select
              value={theme.id}
              onChange={(event) => {
                setTheme(themeById(event.target.value));
              }}
            >
              {THEMES.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.label}
                </option>
              ))}
            </select>
          </label>
        </div>
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
                    style={{ background: clusterColor(theme, cluster) }}
                  />
                  {clusterLabel(cluster, clusterNames)}
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
          ✛ landmark tags
        </label>
      </div>

      <main className="main">
        <div className="map">
          <MapView
            theme={theme}
            conversations={data.conversations}
            landmarks={data.landmarks}
            clusterNames={clusterNames}
            matchedIds={matchedIds}
            hiddenClusters={hiddenClusters}
            showLandmarks={showLandmarks}
            selectedId={selectedId}
            onSelect={handleSelect}
          />
          <TimeBar
            min={dateExtent.min}
            max={dateExtent.max}
            start={dateRange?.start ?? dateExtent.min}
            end={dateRange?.end ?? dateExtent.max}
            active={dateRange !== null}
            onChange={handleDateChange}
            onReset={() => {
              setDateRange(null);
            }}
          />
        </div>
        <DetailPanel
          theme={theme}
          data={data}
          clusterNames={clusterNames}
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
