# frozen_string_literal: true
#
# Generates one /tags/<name>/index.html per unique tag found across the
# `writing` and `talks` collections, using the `tag` layout.
#
# This replaces jekyll-archives, which (as of 2.3) only scans site.posts and
# has no first-class support for custom collections.

module CodeTumbles
  class TagPageGenerator < Jekyll::Generator
    safe true
    priority :low

    COLLECTIONS = %w[writing talks].freeze

    def generate(site)
      tag_index = Hash.new { |h, k| h[k] = [] }

      COLLECTIONS.each do |coll|
        collection = site.collections[coll]
        next unless collection

        collection.docs.each do |doc|
          tags = Array(doc.data["tags"])
          tags.each { |t| tag_index[t.to_s] << doc }
        end
      end

      tag_index.each do |tag, docs|
        site.pages << TagPage.new(site, tag, docs)
      end
    end
  end

  class TagPage < Jekyll::Page
    def initialize(site, tag, docs)
      @site = site
      @base = site.source
      @dir  = File.join("tags", Jekyll::Utils.slugify(tag))
      @name = "index.html"

      process(@name)
      self.data ||= {}
      self.data["layout"] = "tag"
      self.data["title"]  = tag
      self.data["posts"]  = docs.sort_by { |d| d.data["date"] || Time.at(0) }.reverse
      self.data["sitemap"] = false
      self.content = ""
    end
  end
end
