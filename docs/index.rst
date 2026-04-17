.. khisto-python documentation master file, created by
   sphinx-quickstart on Fri Mar  6 17:40:33 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

khisto-python documentation
===========================

Khisto helps you build histograms that look right on the first try.

It replaces fixed-width bin guesses with adaptive bins computed by the Khiops
algorithm, so dense regions stay detailed and sparse regions stay readable.

Pick the entry point that matches your workflow:

- :doc:`demo` for a runnable tour.
- :doc:`array/histogram/index` for a NumPy-like API with better bins.
- :doc:`matplotlib/index` for plots that stay familiar but look sharper.
- :doc:`core/index` for full access to the histogram series.
- :doc:`api_comparison` for a quick comparison with NumPy and matplotlib.

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   Histograms <array/histogram/index>
   Core <core/index>
   Matplotlib <matplotlib/index>

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   API Comparison <api_comparison>
   Demo <demo>

