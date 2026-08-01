"""Talking to Air backend.

Everything the backend owns lives under this one distribution package. It used
to be eleven unqualified top-level names (``api``, ``config``, ``models``,
``services``, ``tools``, ``utils``, ``earthdata_mcp``, ...), which squatted
some of the most generic import names there are and collided for real:
``import earthdata_mcp`` resolved to the sibling harmony-retrieval-mcp *server*
distribution rather than this package's MCP *client*, and only the
``sys.path.insert`` bootstrap at the top of every test file kept the right one
winning. A single namespaced root removes that whole class of problem.
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# The deployment root -- Backend/ in a checkout, /app in the image. The runtime
# volumes (outputs/, overlay_store/, cube_store/, data/) are mounted HERE, one
# level above the package, so anything resolving them must anchor on this
# rather than counting ".." segments up from its own __file__. Counting is what
# broke when this package was introduced: paths written as "../.." from
# tools/satellite_tools/ silently retargeted into tta_backend/ and stopped
# landing on the mounted volumes at all.
APP_ROOT = os.path.dirname(PACKAGE_DIR)
