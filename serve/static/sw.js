const CACHE_PREFIX = 'autoterm-sst-shell-';
const CACHE_NAME = CACHE_PREFIX + 'v1';
const SHELL = [
    '/',
    '/manifest.webmanifest',
    '/static/icons/autoterm-180.png',
    '/static/icons/autoterm-192.png',
    '/static/icons/autoterm-512.png',
    '/static/audio-processor.js',
    '/static/electron-audio-fix.js',
];

self.addEventListener('install', event => {
    event.waitUntil(caches.open(CACHE_NAME).then(cache => cache.addAll(
        SHELL.map(path => new Request(path, { cache: 'reload' }))
    )));
});

self.addEventListener('activate', event => {
    event.waitUntil((async () => {
        const names = await caches.keys();
        await Promise.all(names
            .filter(name => name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME)
            .map(name => caches.delete(name)));
        await self.clients.claim();
    })());
});

self.addEventListener('fetch', event => {
    const request = event.request;
    const { origin, pathname } = new URL(request.url);
    if (request.method !== 'GET' || origin !== self.location.origin || !SHELL.includes(pathname)) {
        return;
    }

    event.respondWith((async () => {
        const cache = await caches.open(CACHE_NAME);
        if (pathname !== '/') {
            return await cache.match(pathname) || fetch(request);
        }

        let response;
        try {
            response = await fetch(request, { cache: 'no-cache' });
        } catch {
            return cache.match('/');
        }
        if (response.ok) {
            await cache.put('/', response.clone());
        }
        return response;
    })());
});
