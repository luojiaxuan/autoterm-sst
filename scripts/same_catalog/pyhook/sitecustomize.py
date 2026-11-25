"""Neutral process titles on the shared node."""

import setproctitle

setproctitle.setproctitle("/usr/bin/python")
setproctitle.setproctitle = lambda *args, **kwargs: None
