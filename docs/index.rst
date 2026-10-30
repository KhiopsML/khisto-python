.. khisto-python documentation master file

===========================================
Khisto — Histograms that fit your data
===========================================

.. rst-class:: hero-tagline

   Drop-in replacements for ``numpy.histogram`` and ``plt.hist`` with
   adaptive, variable-width bins powered by the Khisto algorithm.
   Dense regions get fine bins, sparse regions get wide ones — no tuning needed.

.. container:: binning-comparison

   .. container:: comparison-header

      .. container:: comparison-label

         .. rubric:: Before
            :class: comparison-method

         .. rst-class:: comparison-subtitle

            Equal-width

      .. container:: comparison-label

         .. rubric:: After
            :class: comparison-method comparison-adaptive

         .. rst-class:: comparison-subtitle

            Khisto

   .. rubric:: Lévy
      :class: comparison-heading

   .. container:: comparison-pair

      .. container:: comparison-panel

         .. image:: images/levy-equal-width-light.svg
            :class: comparison-light
            :alt: Before: equal-width bins hide the Lévy peak on logarithmic axes.

         .. image:: images/levy-equal-width-dark.svg
            :class: comparison-dark
            :alt: Before: equal-width bins hide the Lévy peak on logarithmic axes.

      .. container:: comparison-panel

         .. image:: images/levy-khisto-light.svg
            :class: comparison-light
            :alt: After: Khisto reveals the Lévy peak and heavy tail on the same scales.

         .. image:: images/levy-khisto-dark.svg
            :class: comparison-dark
            :alt: After: Khisto reveals the Lévy peak and heavy tail on the same scales.

   .. rubric:: Adult age
      :class: comparison-heading

   .. container:: comparison-pair

      .. container:: comparison-panel

         .. image:: images/adult-equal-width-light.svg
            :class: comparison-light
            :alt: Before: equal-width bins create alternating artificial spikes in integer ages.

         .. image:: images/adult-equal-width-dark.svg
            :class: comparison-dark
            :alt: Before: equal-width bins create alternating artificial spikes in integer ages.

      .. container:: comparison-panel

         .. image:: images/adult-khisto-light.svg
            :class: comparison-light
            :alt: After: Khisto shows the age distribution without alternating binning spikes.

         .. image:: images/adult-khisto-dark.svg
            :class: comparison-dark
            :alt: After: Khisto shows the age distribution without alternating binning spikes.

   .. container:: comparison-caption

      .. raw:: html

         <p>Same data, bin counts and scales.
         Ages: <a class="reference external" href="https://doi.org/10.24432/C5XW20"
         target="_blank" rel="noopener noreferrer"
         aria-label="UCI Adult (opens in a new tab)">UCI Adult</a>.</p>

Get started
-----------

.. admonition:: Density by default
   :class: important

   Unlike NumPy and Matplotlib, Khisto uses density by default.
   :doc:`Learn how to read variable-width histograms <counts_vs_density>`.

.. div:: install-cmd

   .. code-block:: bash

      pip install khisto                # core (NumPy only)
      pip install "khisto[matplotlib]"  # + plotting

.. code-block:: python

   import numpy as np
   import khisto

   data = np.random.normal(0, 1, 10_000)
   hist, bin_edges = khisto.histogram(data)          # optimal bins, no guessing

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: sd-mt-3

   .. grid-item-card:: :octicon:`package;1.5em` NumPy-like API
      :link: histogram/index
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

   .. grid-item-card:: :octicon:`play;1.5em` Interactive demo
      :link: demo
      :link-type: doc

      A runnable notebook tour covering all features.

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: sd-mt-3

   .. grid-item-card:: :octicon:`law;1.5em` Counts or density?
      :link: counts_vs_density
      :link-type: doc

      See why counts mislead when bins have different widths.

   .. grid-item-card:: :octicon:`info;1.5em` Histograms - Khiops
      :class-card: histogram-guide-card sd-card-hover

      Go further, from the simplest histogram to the most complex.

      .. raw:: html

         <a class="sd-stretched-link sd-hide-link-text reference external"
         href="https://khiops.org/learn/histograms/" target="_blank"
         rel="noopener noreferrer"><span>Histograms - Khiops (opens in a new tab)</span></a>

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   Histograms <histogram/index>
   Matplotlib <matplotlib/index>
   Core <core/index>

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   Demo <demo>
   Counts vs density <counts_vs_density>
