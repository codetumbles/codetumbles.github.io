# Code Tumbles

Source for [codetumbles.github.io](https://codetumbles.github.io/).

A minimal custom Jekyll site (Ruby + Bootstrap 5 utilities) built around
multi-collection writing, dark mode, sticky sidebar TOC, Lunr search, and
RSS / sitemap / SEO out of the box.

## Develop locally

```bash
bundle install
bundle exec jekyll serve --livereload
# open http://127.0.0.1:4000/blog/
```

To preview drafts in `_drafts/`:

```bash
bin/serve-drafts
```

## Create a new post

```bash
bin/new-post "My new post title"
# creates _writing/YYYY-MM-DD-my-new-post-title.md with the right front matter
```

To regenerate the default Open Graph image:

```bash
bin/make-og-image
```

## Layout

```
_config.yml          site config + feature flags
_data/navigation.yml main nav links
_writing/            blog posts (date-prefixed filenames, sorted by `date:`)
_talks/              conference talks, podcasts, etc.
_pages/              static pages (about, search, tags, writing, talks)
_includes/           reusable HTML fragments
_layouts/            page templates (default, home, post, page, archive, tag)
assets/css/main.scss site stylesheet (CSS custom properties for theming)
assets/js/           dark mode, TOC, copy-code, anchors, search, progress bar
.github/workflows/   GitHub Actions Pages deploy
```

## Feature flags

Edit `features:` in `_config.yml` to toggle site features. Comments, newsletter,
typewriter, curated lists and Mermaid are off by default.
