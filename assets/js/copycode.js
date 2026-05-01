(function () {
  'use strict';

  if (!navigator.clipboard) return;

  var blocks = document.querySelectorAll('.prose pre');
  blocks.forEach(function (pre) {
    if (pre.querySelector('.copy-code')) return;

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-code';
    btn.setAttribute('aria-label', 'Copy code to clipboard');
    btn.textContent = 'Copy';

    btn.addEventListener('click', function () {
      var code = pre.querySelector('code');
      var text = code ? code.textContent : pre.textContent;
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = 'Copied!';
        btn.classList.add('is-copied');
        setTimeout(function () {
          btn.textContent = 'Copy';
          btn.classList.remove('is-copied');
        }, 1500);
      }).catch(function () {
        btn.textContent = 'Failed';
        setTimeout(function () { btn.textContent = 'Copy'; }, 1500);
      });
    });

    pre.appendChild(btn);
  });
})();
