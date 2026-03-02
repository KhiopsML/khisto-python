# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Root conftest: force a non-interactive matplotlib backend for all tests."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
