"""Crypto news RSS feeds collected by ``cryptosense.news.collect``.

Each feed was fetched once live on 2026-10-24 and returned HTTP 200 with
items that all carried a publication time. Order matters: when the same
article appears in more than one feed in a run, the first feed listed wins.
"""

FEEDS = [
    {"source": "coindesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
    {"source": "cointelegraph", "url": "https://cointelegraph.com/rss"},
    {"source": "decrypt", "url": "https://decrypt.co/feed"},
    {"source": "theblock", "url": "https://www.theblock.co/rss.xml"},
    {"source": "bitcoinmagazine", "url": "https://bitcoinmagazine.com/feed"},
]
