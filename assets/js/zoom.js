document.addEventListener('DOMContentLoaded', function () {
  var images = document.querySelectorAll('.prose img');
  if (images.length === 0) return;

  // Initialize Zooming (an alternative to medium-zoom that allows upscaling)
  var zooming = new Zooming({
    bgColor: getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#ffffff',
    bgOpacity: 0.95,
    customSize: '100%', // Forces the image to zoom as much as possible to fit the screen
    scaleBase: 1,
    zIndex: 999
  });

  zooming.listen(images);

  // Update background color if dark mode is toggled
  var observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.attributeName === 'class') {
        zooming.config({
          bgColor: getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#ffffff'
        });
      }
    });
  });
  
  observer.observe(document.documentElement, { attributes: true });
});
