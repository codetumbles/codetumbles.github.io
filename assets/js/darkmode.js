(function () {
  'use strict';

  var STORAGE_KEY = 'theme';
  var DARK_CLASS = 'dark-mode';
  var LIGHT_THEME_COLOR = '#ffffff';
  var DARK_THEME_COLOR = '#0f1115';

  var root = document.documentElement;
  var meta = document.getElementById('meta-theme-color');
  var btn = document.getElementById('dark-mode-toggle');

  function isDark() {
    return root.classList.contains(DARK_CLASS);
  }

  function syncMetaAndButton() {
    var dark = isDark();
    if (meta) meta.setAttribute('content', dark ? DARK_THEME_COLOR : LIGHT_THEME_COLOR);
    if (btn) btn.setAttribute('aria-pressed', dark ? 'true' : 'false');
  }

  function applyTheme(theme, persist) {
    if (theme === 'dark') {
      root.classList.add(DARK_CLASS);
    } else {
      root.classList.remove(DARK_CLASS);
    }
    syncMetaAndButton();
    if (persist) {
      try { localStorage.setItem(STORAGE_KEY, theme); } catch (e) {}
    }
    document.dispatchEvent(new CustomEvent('themechange', { detail: { theme: theme } }));
  }

  syncMetaAndButton();

  if (btn) {
    btn.addEventListener('click', function () {
      applyTheme(isDark() ? 'light' : 'dark', true);
    });
  }

  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var listener = function (e) {
      try {
        if (localStorage.getItem(STORAGE_KEY)) return;
      } catch (err) {}
      applyTheme(e.matches ? 'dark' : 'light', false);
    };
    if (mq.addEventListener) mq.addEventListener('change', listener);
    else if (mq.addListener) mq.addListener(listener);
  }
})();
