(function () {
  'use strict';

  var bar = document.getElementById('reading-progress');
  if (!bar) return;

  var article = document.getElementById('post-article') || document.querySelector('article.post');
  if (!article) return;

  var ticking = false;
  function update() {
    var rect = article.getBoundingClientRect();
    var total = rect.height - window.innerHeight;
    var scrolled = -rect.top;
    var pct = total > 0 ? Math.max(0, Math.min(1, scrolled / total)) : 0;
    bar.style.width = (pct * 100).toFixed(2) + '%';
    ticking = false;
  }

  window.addEventListener('scroll', function () {
    if (!ticking) {
      window.requestAnimationFrame(update);
      ticking = true;
    }
  }, { passive: true });

  window.addEventListener('resize', update, { passive: true });
  update();
})();
