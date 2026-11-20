// Offline link check of the built site: every internal href/src in dist/ must resolve to a
// file. No network, so it cannot flake. External (http/https/mailto) links are skipped.
import { readdirSync, readFileSync, statSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';

const dist = new URL('../dist/', import.meta.url).pathname;
const BASE = '/mobench';
const files = [];
(function walk(d) {
  for (const f of readdirSync(d)) {
    const p = join(d, f);
    statSync(p).isDirectory() ? walk(p) : files.push(p);
  }
})(dist);

let bad = 0, checked = 0;
for (const file of files.filter((f) => f.endsWith('.html'))) {
  const html = readFileSync(file, 'utf8');
  if (file.includes('/dashboard/')) continue; // generated page; checked as a file, not parsed
  for (const m of html.matchAll(/\b(?:href|src)="([^"]+)"/g)) {
    let url = m[1].split('#')[0].split('?')[0];
    if (!url || /^(https?:|mailto:|data:|javascript:)/.test(url)) continue;
    let target;
    if (url.startsWith('/')) {
      if (!url.startsWith(BASE + '/') && url !== BASE) { console.error(`${file}: absolute link outside base: ${url}`); bad++; continue; }
      target = join(dist, url.slice(BASE.length));
    } else {
      target = join(dirname(file), url);
    }
    checked++;
    if (!(existsSync(target) && (statSync(target).isFile() || existsSync(join(target, 'index.html'))))) {
      console.error(`${file}: broken link ${m[1]}`);
      bad++;
    }
  }
}
// The generated dashboard is not parsed, but it must carry the site credit footer.
const dash = readFileSync(join(dist, 'dashboard/index.html'), 'utf8');
if (!dash.includes('nerfies.github.io') || !dash.includes('Back to project page')) {
  console.error('dashboard is missing the credit footer');
  bad++;
}
console.log(`checked ${checked} internal links in ${files.filter((f) => f.endsWith('.html')).length} pages, ${bad} broken`);
process.exit(bad ? 1 : 0);
