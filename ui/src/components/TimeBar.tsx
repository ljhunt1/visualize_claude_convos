import * as Slider from '@radix-ui/react-slider';

const DAY_MS = 86_400_000;

interface TimeBarProps {
  /** Corpus date extent, ISO dates. */
  min: string;
  max: string;
  /** Current filter range (equals extent when inactive). */
  start: string;
  end: string;
  active: boolean;
  onChange: (start: string, end: string) => void;
  onReset: () => void;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/**
 * Date-range scrubber along the bottom of the map: dual-thumb slider
 * (radix primitive, styled via theme tokens) plus manual date fields.
 */
export function TimeBar({
  min,
  max,
  start,
  end,
  active,
  onChange,
  onReset,
}: TimeBarProps) {
  const minMs = Date.parse(min);
  const nDays = Math.max(1, Math.round((Date.parse(max) - minMs) / DAY_MS));
  const toIso = (day: number) =>
    new Date(minMs + day * DAY_MS).toISOString().slice(0, 10);
  const toDay = (iso: string) =>
    clamp(Math.round((Date.parse(iso) - minMs) / DAY_MS), 0, nDays);

  return (
    <div className="timebar">
      <input
        type="date"
        className="timebar-date"
        value={start}
        min={min}
        max={end}
        aria-label="Start date"
        onChange={(event) => {
          const v = event.target.value;
          if (v) onChange(toIso(toDay(v)), end);
        }}
      />
      <Slider.Root
        className="range-slider"
        min={0}
        max={nDays}
        step={1}
        value={[toDay(start), toDay(end)]}
        onValueChange={(value) => {
          const [s = 0, e = nDays] = value;
          onChange(toIso(s), toIso(e));
        }}
      >
        <Slider.Track className="range-rail">
          <Slider.Range className="range-fill" />
        </Slider.Track>
        <Slider.Thumb className="range-thumb" aria-label="Start date slider" />
        <Slider.Thumb className="range-thumb" aria-label="End date slider" />
      </Slider.Root>
      <input
        type="date"
        className="timebar-date"
        value={end}
        min={start}
        max={max}
        aria-label="End date"
        onChange={(event) => {
          const v = event.target.value;
          if (v) onChange(start, toIso(toDay(v)));
        }}
      />
      <button
        type="button"
        className="timebar-reset"
        disabled={!active}
        onClick={onReset}
      >
        all
      </button>
    </div>
  );
}
