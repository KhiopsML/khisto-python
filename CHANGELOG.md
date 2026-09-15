# Changelog

All notable changes to Khisto are documented in this file.

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

[1.0.2]: https://github.com/KhiopsML/khisto-python/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/KhiopsML/khisto-python/compare/v1.0.0...v1.0.1
