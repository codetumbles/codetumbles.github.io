(function () {
  'use strict';

  var prose = document.querySelector('.prose');
  if (!prose) return;

  var headings = prose.querySelectorAll('h2[id], h3[id], h4[id]');
  headings.forEach(function (h) {
    if (h.querySelector('.anchor-link')) return;
    var a = document.createElement('a');
    a.className = 'anchor-link';
    a.href = '#' + h.id;
    a.textContent = '#';
    a.setAttribute('aria-label', 'Permalink to ' + (h.textContent || '').trim());
    a.addEventListener('click', function (e) {
      if (!navigator.clipboard) return;
      e.preventDefault();
      var url = window.location.origin + window.location.pathname + '#' + h.id;
      navigator.clipboard.writeText(url).catch(function () {});
      history.replaceState(null, '', '#' + h.id);
      h.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    h.appendChild(a);
  });
})();
