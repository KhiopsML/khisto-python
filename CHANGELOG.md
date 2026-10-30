# Changelog

All notable changes  to Khisto are documented in this file for the users.

## Unreleased

### Fixed

- Fix plotting dates and data with measurement units, including axis labels
  and range limits.
- Fix incorrect editor warnings when plotting dates, decimals, or data with
  measurement units.

### Changed

- Add clearer before-and-after examples comparing Khisto with equal-width
  histograms in the documentation.

## [1.0.3] - 2026-10-22

### Changed

- Require Python 3.11 or newer, dropping support for Python 3.10 after its end of life.

### Fixed

- Make very narrow adaptive histogram bins visible by drawing, by default, bar edges in the
  same color as their fill with alpha 0.5. Explicit histogram alpha and edge styling
  are preserved.
- Prevent incorrect histogram frequencies in `hist()` for extreme finite values, including
  values above `1e100` and below `-1e100`, by reusing the frequencies computed by Khisto.
- Prevent `import khisto` from failing when an unrelated `pyproject.toml` is
  present in `site-packages` in a virtual environment.
- Reject `histtype="barstacked"` in `hist()`, which is not supported because
  Khisto accepts only one dataset at a time.
- Explain how to install the Matplotlib extra when plotting without Matplotlib installed.

## [1.0.2] - 2026-09-15

### Added

- Explain counts, densities, and variable-width bins in a dedicated guide.

### Fixed

- Correct Matplotlib plots for observations exactly on internal bin edges. Khiops
  assigns these values to right-closed bins, `(lower, upper]`, while Matplotlib's
  default left-closed convention, `[lower, upper)`, previously shifted them into
  the next bin and produced incorrect counts and densities.
- Prevent `max_bins` from selecting a histogram finer than the best interpretable
  one.

### Changed

- Highlight and update links to the official Khiops histogram documentation.
- Clarify the bin-boundary semantics in the array and Matplotlib API documentation.

## [1.0.1] - 2026-09-07

### Fixed

- Use published documentation URLs so README images render on PyPI.

## [1.0.0] - 2026-09-04

First stable release.

### Added

- Optimal histogram computation powered by the Khiops histogram algorithm.
- NumPy-based histogram results and a Matplotlib-compatible plotting API.
- Support for weighted, density, and cumulative histograms.
- Prebuilt wheels for Linux, macOS, and Windows.
- Public API documentation and usage examples.

[1.0.3]: https://github.com/KhiopsML/khisto-python/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/KhiopsML/khisto-python/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/KhiopsML/khisto-python/compare/v1.0.0...v1.0.1
