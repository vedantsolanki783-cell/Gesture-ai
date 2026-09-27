const CACHE = 'nova-gesture-v1';

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const isNavigation = req.mode === 'navigate';

  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE);

      if (isNavigation) {
        try {
          const fresh = await fetch(req);
          cache.put(req, fresh.clone());
          return fresh;
        } catch {
          const cached = await cache.match(req);
          return cached || cache.match('/index.html');
        }
      }

      const cached = await cache.match(req);
      if (cached) return cached;
      try {
        const fresh = await fetch(req, { mode: req.mode === 'navigate' ? 'same-origin' : 'no-cors' });
        cache.put(req, fresh.clone());
        return fresh;
      } catch (err) {
        throw err;
      }
    })()
  );
});
