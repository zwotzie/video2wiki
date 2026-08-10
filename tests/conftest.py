import video2wiki.downloader as _dl

# yt-dlp rate-limit in Tests deaktivieren (subprocess wird gemockt)
_dl.RATE_LIMIT_DELAY = 0.0
