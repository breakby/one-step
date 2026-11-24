// v3: 같은 출처 파일은 네트워크 우선(옛 버전이 남지 않게), CDN 스크립트·wasm은 캐시 우선(현장에서 네트워크 없이 시작)
const K = 'spaceaudio-v3';
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== K).map(k => caches.delete(k)))).then(() => self.clients.claim())));
const put = (req, r) => { if (r.ok) { const c = r.clone(); caches.open(K).then(x => x.put(req, c)); } return r; };
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const u = new URL(e.request.url);
  if (u.origin === location.origin) e.respondWith(fetch(e.request).then(r => put(e.request, r)).catch(() => caches.match(e.request)));
  else if (u.hostname === 'cdn.jsdelivr.net') e.respondWith(caches.match(e.request).then(m => m || fetch(e.request).then(r => put(e.request, r))));
});
