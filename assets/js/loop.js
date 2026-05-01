(function () {
  'use strict';

  var el = document.getElementById('typewriter');
  if (!el) return;

  // Honor reduced-motion: just print the first string.
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    try {
      var first = JSON.parse(el.dataset.strings || '[]')[0];
      if (first) el.textContent = first;
    } catch (e) {}
    return;
  }

  var strings;
  try { strings = JSON.parse(el.dataset.strings || '[]'); } catch (e) { return; }
  if (!strings || !strings.length) return;

  var TYPE_DELAY = 80;
  var ERASE_DELAY = 40;
  var HOLD_DELAY = 1400;

  var i = 0;
  function play() {
    var s = strings[i % strings.length];
    var c = 0;

    function type() {
      if (c <= s.length) {
        el.textContent = s.slice(0, c);
        c++;
        setTimeout(type, TYPE_DELAY);
      } else {
        setTimeout(erase, HOLD_DELAY);
      }
    }
    function erase() {
      if (c >= 0) {
        el.textContent = s.slice(0, c);
        c--;
        setTimeout(erase, ERASE_DELAY);
      } else {
        i++;
        setTimeout(type, 200);
      }
    }
    type();
  }
  play();
})();
