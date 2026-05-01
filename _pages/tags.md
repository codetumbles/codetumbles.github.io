---
layout: page
title: Tags
permalink: /tags/
sitemap: false
---

{% assign all_tags = "" | split: "" %}
{% for post in site.writing %}
  {% for t in post.tags %}{% assign all_tags = all_tags | push: t %}{% endfor %}
{% endfor %}
{% for post in site.talks %}
  {% for t in post.tags %}{% assign all_tags = all_tags | push: t %}{% endfor %}
{% endfor %}

{% assign unique_tags = all_tags | uniq | sort %}

{% if unique_tags.size == 0 %}
<p class="archive-empty">No tags yet.</p>
{% else %}
<ul class="tag-cloud">
  {% for t in unique_tags %}
    {% assign count = 0 %}
    {% for tag in all_tags %}{% if tag == t %}{% assign count = count | plus: 1 %}{% endif %}{% endfor %}
    {% assign tslug = t | slugify %}
    <li><a href="{{ '/tags/' | append: tslug | append: '/' | relative_url }}">{{ t }} <small>({{ count }})</small></a></li>
  {% endfor %}
</ul>
{% endif %}
