.. khisto-python documentation master file

===========================================
Khisto — Histograms that fit your data
===========================================

.. rst-class:: hero-tagline

   Drop-in replacements for ``numpy.histogram`` and ``plt.hist`` with
   adaptive, variable-width bins powered by the Khisto algorithm.
   Dense regions get fine bins, sparse regions get wide ones — no tuning needed.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Standard Gaussian
      :img-top: images/gaussian-quick-start.png

      Bins concentrate around the interesting areas — exactly
      matching the density of a normal distribution.

   .. grid-item-card:: Heavy-tailed Pareto
      :img-top: images/pareto-quick-start.png

      Log-log axes reveal how adaptive bins track a power-law decay
      over four orders of magnitude.

Get started
-----------

.. div:: install-cmd

   .. code-block:: bash

      pip install khisto            # core (NumPy only)
      pip install "khisto[matplotlib]"  # + plotting

.. code-block:: python

   import numpy as np
   from khisto import histogram

   data = np.random.normal(0, 1, 10_000)
   hist, bin_edges = histogram(data)          # optimal bins, no guessing

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: sd-mt-3

   .. grid-item-card:: :octicon:`package;1.5em` NumPy-like API
      :link: array/histogram/index
      :link-type: doc

      ``histogram(data)`` returns ``(hist, bin_edges)`` — same shape as
      ``numpy.histogram``, better bins.

   .. grid-item-card:: :octicon:`graph;1.5em` Matplotlib integration
      :link: matplotlib/index
      :link-type: doc

      ``khisto.matplotlib.hist`` plots like ``plt.hist`` with density,
      cumulative, step, and log-scale support.

   .. grid-item-card:: :octicon:`telescope;1.5em` Core engine
      :link: core/index
      :link-type: doc

      ``compute_histograms`` exposes every granularity level so you can
      pick the resolution that suits your analysis.

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: sd-mt-1

   .. grid-item-card:: :octicon:`git-compare;1.5em` API comparison
      :link: api_comparison
      :link-type: doc

      Side-by-side parameter tables for NumPy, Matplotlib, and Khisto.

   .. grid-item-card:: :octicon:`play;1.5em` Interactive demo
      :link: demo
      :link-type: doc

      A runnable notebook tour covering all features.

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

