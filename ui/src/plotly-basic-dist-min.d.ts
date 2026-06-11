/**
 * plotly.js-basic-dist-min ships no types; it has the same API surface as
 * plotly.js (minus the trace types excluded from the basic bundle), so
 * borrow the full typings from @types/plotly.js.
 */
declare module 'plotly.js-basic-dist-min' {
  import * as Plotly from 'plotly.js';

  export = Plotly;
}
