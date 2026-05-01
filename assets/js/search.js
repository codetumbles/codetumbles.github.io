(function () {
  'use strict';

  var input = document.getElementById('search-input');
  var results = document.getElementById('search-results');
  if (!input || !results) return;

  var indexUrl = (document.documentElement.dataset.baseurl || '') + '/search.json';
  var docs = [];
  var idx = null;

  function buildIndex(data) {
    docs = data;
    idx = lunr(function () {
      this.ref('id');
      this.field('title', { boost: 10 });
      this.field('tags',  { boost: 5 });
      this.field('content');
      var self = this;
      data.forEach(function (d) { self.add(d); });
    });
  }

  function render(matches) {
    results.innerHTML = '';
    if (matches.length === 0) {
      results.innerHTML = '<li>No results.</li>';
      return;
    }
    matches.slice(0, 20).forEach(function (m) {
      var li = document.createElement('li');
      var a = document.createElement('a');
      a.href = m.url;
      a.textContent = m.title;
      li.appendChild(a);
      var snippet = document.createElement('span');
      snippet.className = 'search-snippet';
      snippet.textContent = (m.content || '').slice(0, 160) + (m.content && m.content.length > 160 ? '...' : '');
      li.appendChild(snippet);
      results.appendChild(li);
    });
  }

  function search(query) {
    if (!idx || !query) { results.innerHTML = ''; return; }
    var matches;
    try {
      matches = idx.search(query);
    } catch (e) {
      matches = idx.search(query.replace(/[^a-z0-9 ]/gi, ' ').trim());
    }
    var hits = matches.map(function (m) {
      var doc = docs[parseInt(m.ref, 10) - 1];
      return doc;
    }).filter(Boolean);
    render(hits);
  }

  // The base URL might not be on documentElement; fall back to Jekyll's relative_url at build time.
  // We use a custom data attribute set below.
  var baseEl = document.querySelector('meta[name="site-baseurl"]');
  if (baseEl) indexUrl = baseEl.content + '/search.json';

  fetch(indexUrl).then(function (r) { return r.json(); }).then(function (data) {
    buildIndex(data);
    if (input.value.trim()) search(input.value.trim());
  }).catch(function () {
    results.innerHTML = '<li>Search index failed to load.</li>';
  });

  var t = null;
  input.addEventListener('input', function () {
    clearTimeout(t);
    t = setTimeout(function () { search(input.value.trim()); }, 120);
  });
})();
