(function () {
  'use strict';

  var body = document.getElementById('post-body');
  if (!body) return;

  var navs = [
    document.getElementById('toc-desktop'),
    document.getElementById('toc-mobile')
  ].filter(Boolean);
  if (navs.length === 0) return;

  function readRange(nav) {
    var min = parseInt(nav.getAttribute('data-min-level'), 10) || 2;
    var max = parseInt(nav.getAttribute('data-max-level'), 10) || 3;
    return [Math.min(min, max), Math.max(min, max)];
  }

  var range = readRange(navs[0]);
  var selectors = [];
  for (var i = range[0]; i <= range[1]; i++) selectors.push('h' + i);
  var headings = Array.prototype.slice.call(body.querySelectorAll(selectors.join(',')))
    .filter(function (h) { return h.id; });
  if (headings.length === 0) return;

  function headingText(heading) {
    var clone = heading.cloneNode(true);
    Array.prototype.slice.call(clone.querySelectorAll('.anchor-link')).forEach(function (anchor) {
      anchor.remove();
    });
    return clone.textContent.trim();
  }

  function buildList(headings, minLevel) {
    var list = document.createElement('ol');
    list.style.margin = 0;
    list.style.padding = 0;
    list.style.listStyle = 'none';

    var stacks = [{ level: minLevel - 1, list: list }];

    headings.forEach(function (h) {
      var level = parseInt(h.tagName.substring(1), 10);

      while (stacks.length > 1 && level <= stacks[stacks.length - 1].level) stacks.pop();

      while (stacks[stacks.length - 1].level < level - 1) {
        var lastLi = stacks[stacks.length - 1].list.lastElementChild;
        if (!lastLi) {
          lastLi = document.createElement('li');
          stacks[stacks.length - 1].list.appendChild(lastLi);
        }
        var nestedList = document.createElement('ol');
        nestedList.style.listStyle = 'none';
        nestedList.style.paddingLeft = '0.8rem';
        lastLi.appendChild(nestedList);
        stacks.push({ level: stacks[stacks.length - 1].level + 1, list: nestedList });
      }

      if (level > stacks[stacks.length - 1].level) {
        var parentLi = stacks[stacks.length - 1].list.lastElementChild;
        if (parentLi) {
          var nested = document.createElement('ol');
          nested.style.listStyle = 'none';
          nested.style.paddingLeft = '0.8rem';
          parentLi.appendChild(nested);
          stacks.push({ level: level, list: nested });
        }
      }

      var li = document.createElement('li');
      var a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = headingText(h);
      a.setAttribute('data-target-id', h.id);
      li.appendChild(a);
      stacks[stacks.length - 1].list.appendChild(li);
    });

    return list;
  }

  navs.forEach(function (nav) {
    var range = readRange(nav);
    var min = range[0];
    nav.appendChild(buildList(headings, min));
  });

  // Scroll-spy via IntersectionObserver
  var allLinks = navs.reduce(function (acc, nav) {
    return acc.concat(Array.prototype.slice.call(nav.querySelectorAll('a[data-target-id]')));
  }, []);
  var linksByTarget = {};
  allLinks.forEach(function (a) {
    var id = a.getAttribute('data-target-id');
    (linksByTarget[id] = linksByTarget[id] || []).push(a);
  });

  var visible = new Set();
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) visible.add(entry.target.id);
      else visible.delete(entry.target.id);
    });

    allLinks.forEach(function (a) { a.classList.remove('is-active'); });

    var activeId = null;
    for (var i = 0; i < headings.length; i++) {
      if (visible.has(headings[i].id)) { activeId = headings[i].id; break; }
    }
    if (!activeId) {
      var scrollY = window.scrollY;
      for (var j = headings.length - 1; j >= 0; j--) {
        if (headings[j].getBoundingClientRect().top + window.scrollY <= scrollY + 120) {
          activeId = headings[j].id;
          break;
        }
      }
    }
    if (activeId && linksByTarget[activeId]) {
      linksByTarget[activeId].forEach(function (a) { a.classList.add('is-active'); });
    }
  }, { rootMargin: '-80px 0px -70% 0px', threshold: 0 });

  headings.forEach(function (h) { observer.observe(h); });
})();
