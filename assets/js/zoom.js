document.addEventListener('DOMContentLoaded', function () {
  // Only zoom images inside the main prose content
  var images = document.querySelectorAll('.prose img');
  if (images.length === 0) return;

  // Initialize Medium Zoom
  var zoom = mediumZoom(images, {
    margin: 24,
    background: getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#ffffff',
    scrollOffset: 40
  });

  // Update background color if dark mode is toggled
  var observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.attributeName === 'class') {
        zoom.update({
          background: getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#ffffff'
        });
      }
    });
  });
  
  observer.observe(document.documentElement, { attributes: true });
});
