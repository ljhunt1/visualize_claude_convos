import { useMemo } from 'react';
import { clusterColor, clusterLabel, type Theme } from 'themes';
import type { Conversation, UIData } from 'types';

interface DetailPanelProps {
  theme: Theme;
  data: UIData;
  clusterNames: ReadonlyMap<number, string>;
  selected: Conversation | null;
  onClose: () => void;
}

function SpecimenCard({
  theme,
  conversation,
  clusterNames,
  onClose,
}: {
  theme: Theme;
  conversation: Conversation;
  clusterNames: ReadonlyMap<number, string>;
  onClose: () => void;
}) {
  const tags = useMemo(
    () => [...conversation.tags].sort((a, b) => b.score - a.score),
    [conversation.tags],
  );
  const ink = clusterColor(theme, conversation.cluster);

  return (
    <article className="specimen">
      <header className="specimen-head">
        <div className="specimen-kicker">
          <span className="cluster-dot" style={{ background: ink }} />
          <span>{clusterLabel(conversation.cluster, clusterNames)}</span>
          <span className="kicker-sep">·</span>
          <span>{conversation.date}</span>
          <span className="kicker-sep">·</span>
          <span>{conversation.nTurns} turns</span>
          <span className="kicker-sep">·</span>
          <span>{conversation.nWords.toLocaleString()} words</span>
        </div>
        <button
          type="button"
          className="close-button"
          onClick={onClose}
          aria-label="Close detail panel"
        >
          ✕
        </button>
      </header>

      <h2 className="specimen-title">{conversation.title}</h2>

      <p className="specimen-summary">{conversation.summary}</p>

      <p className="specimen-link">
        <a href={conversation.url} target="_blank" rel="noreferrer">
          open in claude.ai ↗
        </a>
      </p>

      <section>
        <h3 className="section-label">Tags</h3>
        <ul className="tag-list">
          {tags.map((tag) => (
            <li key={tag.name} className="tag-row">
              <span className="tag-name">{tag.name}</span>
              <span className="tag-bar-track">
                <span
                  className="tag-bar"
                  style={{
                    width: `${String(Math.round(tag.score * 100))}%`,
                    background: ink,
                  }}
                />
              </span>
              <span className="tag-score">{tag.score.toFixed(2)}</span>
            </li>
          ))}
        </ul>
      </section>

      <details className="transcript">
        <summary className="section-label">Transcript</summary>
        <pre className="transcript-text">{conversation.transcript}</pre>
      </details>
    </article>
  );
}

function AtlasOverview({
  theme,
  data,
  clusterNames,
}: {
  theme: Theme;
  data: UIData;
  clusterNames: ReadonlyMap<number, string>;
}) {
  const { conversations, landmarks, meta } = data;

  const clusterCounts = useMemo(() => {
    const counts = new Map<number | null, number>();
    for (const conv of conversations) {
      counts.set(conv.cluster, (counts.get(conv.cluster) ?? 0) + 1);
    }
    return [...counts.entries()].sort(
      ([a], [b]) =>
        (a ?? Number.MAX_SAFE_INTEGER) - (b ?? Number.MAX_SAFE_INTEGER),
    );
  }, [conversations]);

  const dates = conversations.map((c) => c.date).sort();
  const first = dates[0];
  const last = dates[dates.length - 1];

  return (
    <div className="overview">
      <p className="overview-lede">
        {conversations.length} conversations and {landmarks.length} landmark
        tags, arranged by semantic similarity. Click a dot to read one; the ✛
        marks are reference tags, not conversations.
      </p>

      <section>
        <h3 className="section-label">Clusters</h3>
        <ul className="cluster-list">
          {clusterCounts.map(([cluster, count]) => (
            <li key={cluster ?? 'noise'} className="cluster-row">
              <span
                className="cluster-dot"
                style={{ background: clusterColor(theme, cluster) }}
              />
              <span className="cluster-name">
                {clusterLabel(cluster, clusterNames)}
              </span>
              <span className="cluster-count">{count}</span>
            </li>
          ))}
        </ul>
      </section>

      {first && last && (
        <section>
          <h3 className="section-label">Range</h3>
          <p className="overview-range">
            {first} — {last}
          </p>
        </section>
      )}

      <footer className="overview-meta">
        <p>
          generated {meta.generatedAt}
          <br />
          labels: {meta.labelModel} (prompt {meta.labelPromptFp})
          <br />
          embeddings: {meta.embedModel}
          <br />
          umap n_neighbors={meta.umap.nNeighbors} min_dist={meta.umap.minDist}
          <br />
          {meta.clusterAlgo}
        </p>
      </footer>
    </div>
  );
}

export function DetailPanel({
  theme,
  data,
  clusterNames,
  selected,
  onClose,
}: DetailPanelProps) {
  return (
    <aside className="panel">
      {selected ? (
        <SpecimenCard
          theme={theme}
          conversation={selected}
          clusterNames={clusterNames}
          onClose={onClose}
        />
      ) : (
        <AtlasOverview theme={theme} data={data} clusterNames={clusterNames} />
      )}
    </aside>
  );
}
