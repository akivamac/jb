/* Shared sidebar - renders the shell markup and fills data.
   Used by both index.html and tools.html so the two pages cannot drift.

   Sidebar.mount()          inject the shell (nav / stats / expert list)
   Sidebar.setActive(page)  highlight the nav item for a page
   Sidebar.renderData(...)  fill expert list + stats
   Sidebar.load(apiBase)    fetch + render (for pages with no SSE of their own) */
(function (global) {
  'use strict';

  function escHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function escAttr(s) {
    return escHtml(s).replace(/`/g, '&#96;');
  }

  /* Outline (stroke) icons - no filled/emoji glyphs. */
  var ICONS = {
    home: '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    tools: '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    stats: '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'
  };

  function icon(name) {
    return '<svg class="nav-icon" width="15" height="15" viewBox="0 0 24 24" fill="none" ' +
      'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" ' +
      'aria-hidden="true" focusable="false">' + (ICONS[name] || '') + '</svg>';
  }

  var NAV = [
    { page: 'home',  href: 'index.html',      icon: 'home',  label: 'Home' },
    { page: 'tools', href: 'tools.html',      icon: 'tools', label: 'Tools' },
    { page: 'stats', href: 'index.html#/stats', icon: 'stats', label: 'Stats' }
  ];

  function shell(active) {
    var nav = NAV.map(function (n) {
      var on = n.page === active;
      return '<a class="sidebar-nav-item' + (on ? ' is-active' : '') + '" href="' + n.href + '"' +
        (on ? ' aria-current="page"' : '') + ' data-page="' + n.page + '">' +
        icon(n.icon) + '<span class="nav-label">' + escHtml(n.label) + '</span></a>';
    }).join('');

    return '<nav class="sidebar-nav">' + nav + '</nav>';
  }

  function root() {
    return document.getElementById('sidebar');
  }

  /* Inject the shell. active = 'home' | 'tools'; omit to infer from location. */
  function mount(active) {
    if (!root()) return null;
    if (!active) {
      active = /tools\.html$/.test(global.location.pathname) ? 'tools' : 'home';
      if (/#\/stats/.test(global.location.hash || '')) active = 'stats';
    }
    root().innerHTML = shell(active);
    return root();
  }

  function setActive(page) {
    document.querySelectorAll('.sidebar-nav-item').forEach(function (el) {
      var on = el.getAttribute('data-page') === page;
      el.classList.toggle('is-active', on);
      if (on) el.setAttribute('aria-current', 'page');
      else el.removeAttribute('aria-current');
    });
  }

  function setText(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  /* opts: {experts, fleetStats, activeName, onSelect} */
  function renderData(opts) {
    opts = opts || {};
    var experts = opts.experts || [];
    var stats = opts.fleetStats || {};
    var list = document.getElementById('expert-list');
    if (!list) return;

    list.innerHTML = experts.map(function (e) {
      var cls = e.running ? 'running' : (e.queued ? 'queued' : 'idle');
      var label = e.running ? 'running' : (e.queued ? 'queued' : 'idle');
      var active = opts.activeName && opts.activeName === e.name ? ' active' : '';
      var attr = opts.onSelect ? ' data-expert="' + escAttr(e.name) + '"' : '';
      return '<li class="expert-item' + active + '"' + attr + ' tabindex="0">' +
        '<span class="name">' + escHtml(e.name) + '</span>' +
        '<span class="status-tag ' + cls + '">' + label + '</span>' +
      '</li>';
    }).join('');

    if (opts.onSelect) {
      list.querySelectorAll('.expert-item').forEach(function (li) {
        var name = li.getAttribute('data-expert');
        li.addEventListener('click', function () { opts.onSelect(name); });
        li.addEventListener('keydown', function (ev) {
          if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); opts.onSelect(name); }
        });
      });
    }

    setText('expert-count', String(experts.length));
    setText('sb-running', String(experts.filter(function (e) { return e.running; }).length));
    setText('sb-total', String(experts.length));
    setText('sb-steps', (stats.total_step || 0).toLocaleString());
    setText('sb-loss', stats.avg_loss != null ? Number(stats.avg_loss).toFixed(4) : '—');
  }

  /* For pages without their own SSE (tools.html). */
  function load(apiBase, opts) {
    apiBase = apiBase || '';
    return fetch(apiBase + '/api/experts', { cache: 'no-store' })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (data) {
        renderData({
          experts: data.experts || [],
          fleetStats: {
            total_step: data.total_step,
            avg_loss: data.avg_loss,
            running_count: data.running_count
          },
          onSelect: opts && opts.onSelect
        });
      })
      .catch(function (e) { console.error('Sidebar.load failed:', e); });
  }

  global.Sidebar = { mount: mount, setActive: setActive, renderData: renderData, load: load };
})(window);
