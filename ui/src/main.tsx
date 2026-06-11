import '@fontsource-variable/archivo/index.css';
import '@fontsource-variable/cormorant/index.css';
import '@fontsource-variable/fraunces/index.css';
import '@fontsource-variable/shantell-sans/index.css';
import '@fontsource/ibm-plex-mono/400.css';
import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/500.css';
import App from 'App';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';

const rootEl = document.getElementById('root');
if (!rootEl) throw new Error('#root not found');

createRoot(rootEl).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
