source "https://rubygems.org"

gem "jekyll", "~> 4.3"

group :jekyll_plugins do
  gem "jekyll-feed",     "~> 0.17"
  gem "jekyll-sitemap",  "~> 1.4"
  gem "jekyll-seo-tag",  "~> 2.8"
  gem "jemoji",          "~> 0.13"
end

# kramdown GFM parser (required for `kramdown.input: GFM`)
gem "kramdown-parser-gfm", "~> 1.1"

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :install_if => Gem.win_platform?

# Lock http_parser.rb gem to v0.6.x on JRuby builds
gem "http_parser.rb", "~> 0.6.0", :install_if => Proc.new { RUBY_ENGINE == "jruby" }
