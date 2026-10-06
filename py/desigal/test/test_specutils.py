# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""Tests for :mod:`desigal.specutils`.

These exercise the spectral utilities on small synthetic inputs, so the whole
module runs in a few seconds and needs no DESI data files, no network and no
database. Tests that genuinely require the SFD dust maps are skipped unless
``$DUST_DIR`` is set.

Two things are covered deliberately:

* **Regressions.** Every bug fixed in issues #36, #38, #40, #42, #44, #46 and
  #48 has a test here that fails against the code as it was.
* **Invariants.** Flux conservation, redshift scalings, inverse-variance
  addition across cameras and normalization fixed points -- the properties
  that should hold whatever the implementation does internally.

A final ``TestKnownBugs`` class records defects that are still open, as
``expectedFailure``. Those are not noise: when one of them starts passing,
unittest reports an unexpected success, which is the signal that the
corresponding issue can be closed.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from astropy.table import Table

from ..specutils.coadd_cameras import coadd_cameras
from ..specutils.coaddition import coadd_flux, weighted_quantiles
from ..specutils.normalize import normalize
from ..specutils.redshift import deredshift
from ..specutils.resample import resample

# The private resampling helpers are imported inside the tests that use them,
# not here: a missing symbol at module scope makes the whole file uncollectable,
# which would hide every other failure behind one import error.

# np.trapz was renamed in numpy 2.0.
_trapz = getattr(np, "trapezoid", None) or np.trapz

#: DESI-like per-camera wavelength grids, with realistic b/r and r/z overlaps.
CAM_WAVE = {
    "b": np.arange(3600.0, 5800.0, 0.8),
    "r": np.arange(5760.0, 7620.0, 0.8),
    "z": np.arange(7520.0, 9824.0, 0.8),
}

HAVE_DUST = bool(os.environ.get("DUST_DIR"))


def camera_dicts(nspec=4, seed=0):
    """Synthetic per-camera flux/wave/ivar/mask dictionaries."""
    rng = np.random.default_rng(seed)
    flux = {b: rng.normal(10.0, 1.0, (nspec, len(w))) for b, w in CAM_WAVE.items()}
    ivar = {b: np.full((nspec, len(w)), 4.0) for b, w in CAM_WAVE.items()}
    mask = {b: np.zeros((nspec, len(w)), dtype=np.uint32) for b, w in CAM_WAVE.items()}
    return flux, dict(CAM_WAVE), ivar, mask


def fibermap(nspec=4):
    """Minimal fibermap carrying the columns stack_spectra needs."""
    return Table(
        {
            "TARGETID": np.arange(nspec),
            "TARGET_RA": np.full(nspec, 180.0),
            "TARGET_DEC": np.full(nspec, 20.0),
        }
    )


class TestPackageImports(unittest.TestCase):
    """Import-time behaviour. Regression for #36 and #46."""

    def test_specutils_imports(self):
        """The package imports and exposes its public entry points (#36)."""
        from .. import specutils

        for name in (
            "coadd_cameras",
            "coadd_flux",
            "deredshift",
            "get_sky",
            "get_spectra",
            "mw_dust_correct",
            "normalize",
            "resample",
            "stack_spectra",
        ):
            self.assertTrue(hasattr(specutils, name), "missing export: " + name)

    def test_import_without_dust_dir(self):
        """Importing must not require $DUST_DIR (#46).

        SFDMap() used to be constructed at module scope, which made the whole
        package unimportable on machines without the dust maps -- even for
        code paths that never touch dust correction.
        """
        env = dict(os.environ)
        env.pop("DUST_DIR", None)
        result = subprocess.run(
            [sys.executable, "-c", "import desigal.specutils"],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode, 0, "import failed without $DUST_DIR:\n" + result.stderr
        )


class TestDeredshift(unittest.TestCase):
    """Redshift corrections. Regression for #40."""

    def setUp(self):
        self.flux = np.ones((3, 10))

    def test_accepts_scalar_list_and_array(self):
        """z_in may be a float, list, ndarray or astropy Column (#40)."""
        column = Table({"Z": np.array([0.1, 0.2, 0.3])})["Z"]
        for label, z_in in (
            ("ndarray", np.array([0.1, 0.2, 0.3])),
            ("Column", column),
            ("np.float64", np.float64(0.2)),
            ("float", 0.2),
            ("list", [0.1, 0.2, 0.3]),
        ):
            with self.subTest(z_in=label):
                out = deredshift(self.flux, z_in, 0.0, "flux")
                self.assertEqual(out.shape, (3, 10))

    def test_scalar_matches_one_element_array(self):
        """A scalar z_in must agree exactly with the 1-element array form."""
        scalar = deredshift(np.ones((1, 4)), 0.37, 0.0, "flux")
        array = deredshift(np.ones((1, 4)), np.array([0.37]), 0.0, "flux")
        np.testing.assert_array_equal(scalar, array)

    def test_array_z_out_broadcasts(self):
        """A per-object z_out must not broadcast into a square matrix (#40)."""
        out = deredshift(self.flux, np.array([0.1, 0.2, 0.3]), np.zeros(3), "flux")
        self.assertEqual(out.shape, (3, 10))

    def test_scalings(self):
        """Flux, wavelength and ivar carry the right powers of (1 + z)."""
        z = np.array([1.0, 3.0])
        ones = np.ones((2, 1))
        np.testing.assert_allclose(
            deredshift(ones, z, 0.0, "flux").ravel(), [2.0, 4.0]
        )
        np.testing.assert_allclose(
            deredshift(ones, z, 0.0, "wave").ravel(), [0.5, 0.25]
        )
        np.testing.assert_allclose(
            deredshift(ones, z, 0.0, "ivar").ravel(), [0.25, 1.0 / 16.0]
        )

    def test_ivar_consistent_with_flux(self):
        """ivar must scale as the inverse square of the flux scaling.

        If f -> f * a then var -> var * a**2, so ivar -> ivar / a**2 and the
        product f**2 * ivar is invariant.
        """
        z = np.array([0.4, 1.7])
        ones = np.ones((2, 1))
        f = deredshift(ones, z, 0.0, "flux")
        i = deredshift(ones, z, 0.0, "ivar")
        np.testing.assert_allclose(f**2 * i, np.ones((2, 1)))

    def test_round_trip(self):
        """Going to rest frame and back returns the original."""
        z = np.array([0.3, 0.9])
        flux = np.full((2, 5), 7.0)
        rest = deredshift(flux, z, 0.0, "flux")
        back = deredshift(rest, np.zeros(2), z, "flux")
        np.testing.assert_allclose(back, flux)

    def test_dict_input_returns_dict(self):
        """Per-camera dictionaries are handled key by key."""
        data = {"b": np.ones((2, 4)), "r": np.ones((2, 4))}
        out = deredshift(data, np.array([1.0, 3.0]), 0.0, "flux")
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), {"b", "r"})
        for value in out.values():
            self.assertEqual(value.shape, (2, 4))


class TestCoaddCameras(unittest.TestCase):
    """Camera coaddition. Regression for #44."""

    def setUp(self):
        self.flux, self.wave, self.ivar, self.mask = camera_dicts(nspec=2)
        for band in self.flux:
            self.flux[band][:] = 1.0
            self.ivar[band][:] = 1.0

    def test_returns_three_without_mask_four_with(self):
        """The arity of the return value depends on whether a mask is given.

        stack_spectra unpacked three values unconditionally, which broke for
        any masked Spectra (#44). Unconditionally unpacking four breaks the
        unmasked path instead, so both arities are pinned here.
        """
        self.assertEqual(len(coadd_cameras(self.flux, self.wave, self.ivar)), 3)
        self.assertEqual(
            len(coadd_cameras(self.flux, self.wave, self.ivar, self.mask)), 4
        )

    def test_output_grid_is_sorted_and_spans_all_cameras(self):
        _, wave, _ = coadd_cameras(self.flux, self.wave, self.ivar)
        self.assertTrue(np.all(np.diff(wave) > 0), "output grid not increasing")
        self.assertAlmostEqual(wave[0], CAM_WAVE["b"][0])
        self.assertLessEqual(wave[-1], CAM_WAVE["z"][-1])

    def test_constant_flux_is_preserved(self):
        """Coadding identical flat spectra must return the same flat value."""
        flux, _, _ = coadd_cameras(self.flux, self.wave, self.ivar)
        np.testing.assert_allclose(flux, 1.0)

    def test_ivar_adds_in_camera_overlap(self):
        """Inverse variances add where two cameras overlap."""
        _, wave, ivar = coadd_cameras(self.flux, self.wave, self.ivar)
        blue_only = wave < 5750.0
        overlap = (wave > 5765.0) & (wave < 5795.0)
        np.testing.assert_allclose(np.unique(ivar[:, blue_only]), [1.0])
        np.testing.assert_allclose(np.unique(ivar[:, overlap]), [2.0])

    def test_mask_zeroes_ivar(self):
        """Masked pixels must end up with zero inverse variance."""
        self.mask["b"][:, :100] = 1
        _, _, ivar, _ = coadd_cameras(self.flux, self.wave, self.ivar, self.mask)
        np.testing.assert_allclose(ivar[:, :100], 0.0)
        self.assertTrue(np.all(ivar[:, 100:110] > 0))


class TestNormalize(unittest.TestCase):
    """Normalization methods. Regression for #38."""

    def setUp(self):
        self.wave = np.linspace(4000.0, 6000.0, 500)
        self.flat = np.ones((3, 500))
        self.ivar = np.full((3, 500), 4.0)

    def test_median_normalizes_flat_to_one(self):
        flux, _ = normalize(
            self.wave, self.flat.copy(), self.ivar.copy(), method="median"
        )
        np.testing.assert_allclose(flux, 1.0)

    def test_mean_normalizes_flat_to_one(self):
        flux, _ = normalize(
            self.wave, self.flat.copy(), self.ivar.copy(), method="mean"
        )
        np.testing.assert_allclose(flux, 1.0)

    def test_flux_window_runs_and_normalizes_flat_to_one(self):
        """scipy.integrate.simps was removed in SciPy 1.14 (#38)."""
        flux, _ = normalize(
            self.wave,
            self.flat.copy(),
            self.ivar.copy(),
            method="flux-window",
            flux_window=(4500.0, 5500.0),
        )
        np.testing.assert_allclose(flux, 1.0)

    def test_scaled_spectrum_normalizes_back(self):
        """A spectrum scaled by a constant normalizes to the same result."""
        scaled = self.flat * 17.0
        flux, _ = normalize(self.wave, scaled, self.ivar.copy(), method="median")
        np.testing.assert_allclose(flux, 1.0)

    def test_ivar_rescaled_by_norm_squared(self):
        """Normalizing flux by n must scale ivar by n**2."""
        flux_in = self.flat * 4.0
        _, ivar_out = normalize(
            self.wave, flux_in.copy(), self.ivar.copy(), method="median"
        )
        np.testing.assert_allclose(ivar_out, self.ivar * 16.0)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            normalize(self.wave, self.flat.copy(), self.ivar.copy(), method="nope")


class TestResample(unittest.TestCase):
    """Resampling methods. Regression for #48."""

    def setUp(self):
        self.wave = np.arange(4000.0, 6000.0, 0.8)
        self.n = len(self.wave)
        # A Gaussian emission line on a flat continuum, well inside the grid.
        self.model = 1.0 + 5.0 * np.exp(
            -0.5 * ((self.wave - 5000.0) / 8.0) ** 2
        )
        self.wave_new = np.arange(4100.0, 5900.0, 1.6)

    def _tiled(self, nspec=3):
        flux = np.tile(self.model, (nspec, 1))
        ivar = np.full((nspec, self.n), 4.0)
        return np.tile(self.wave, (nspec, 1)), flux, ivar

    def test_all_methods_run(self):
        """linear, flux-cons and sn-cons all produce the right shape (#48)."""
        wave, flux, ivar = self._tiled()
        for method in ("linear", "flux-cons", "sn-cons"):
            with self.subTest(method=method):
                out_flux, out_ivar = resample(
                    self.wave_new, wave, flux.copy(), ivar.copy(),
                    method=method, n_workers=1,
                )
                self.assertEqual(out_flux.shape, (3, len(self.wave_new)))
                self.assertEqual(out_ivar.shape, (3, len(self.wave_new)))

    def test_unknown_method_raises(self):
        wave, flux, ivar = self._tiled()
        with self.assertRaises(ValueError):
            resample(self.wave_new, wave, flux, ivar, method="nope", n_workers=1)

    def test_project_row_sums_equal_bin_width_ratio(self):
        """Each output bin draws exactly its own width from the input grid."""
        from ..specutils.resample import _project

        for step in (0.8, 1.6, 0.4):
            with self.subTest(step=step):
                rows = _project(self.wave, np.arange(4100.0, 5900.0, step)).sum(axis=1)
                np.testing.assert_allclose(rows, step / 0.8, rtol=1e-10)

    def test_sn_cons_identity_resample(self):
        """Resampling onto the same grid must return the input."""
        from ..specutils.resample import _sn_conserving_resample

        out = _sn_conserving_resample(self.wave, self.model.copy(), self.wave.copy())
        np.testing.assert_allclose(out, self.model, atol=1e-12)

    def test_sn_cons_conserves_flux(self):
        """The integral under the spectrum survives rebinning."""
        from ..specutils.resample import _sn_conserving_resample

        out = _sn_conserving_resample(self.wave, self.model.copy(), self.wave_new)
        inside = (self.wave >= self.wave_new[0]) & (self.wave <= self.wave_new[-1])
        before = _trapz(self.model[inside], self.wave[inside])
        after = _trapz(out, self.wave_new)
        self.assertAlmostEqual(after / before, 1.0, places=2)

    def test_sn_cons_agrees_with_flux_cons_on_smooth_data(self):
        """On smooth, uniformly weighted data the two methods coincide.

        This is a consistency check, not a demonstration of the S/N advantage
        claimed in the docstring -- that only shows up on sharp features.
        """
        from ..specutils.resample import _sn_conserving_resample

        sn = _sn_conserving_resample(self.wave, self.model.copy(), self.wave_new)
        wave, flux, ivar = self._tiled(nspec=1)
        fc, _ = resample(
            self.wave_new, wave, flux, ivar, method="flux-cons", n_workers=1
        )
        good = np.isfinite(fc[0]) & np.isfinite(sn)
        np.testing.assert_allclose(sn[good], fc[0][good], rtol=1e-9)

    def test_sn_cons_zero_ivar_does_not_produce_nan(self):
        """A masked input pixel must not poison the whole output ivar.

        1/ivar is inf for a no-data pixel, and 0 * inf = nan for every zero
        entry of the projection matrix -- which is most of it. A single masked
        pixel used to turn almost the entire output ivar into nan.
        """
        from ..specutils.resample import _sn_conserving_resample

        ivar = np.full(self.n, 4.0)
        ivar[1000] = 0.0
        _, out_ivar = _sn_conserving_resample(
            self.wave, self.model.copy(), self.wave_new, ivar=ivar
        )
        self.assertFalse(np.any(np.isnan(out_ivar)), "nan in resampled ivar")
        self.assertFalse(np.any(np.isinf(out_ivar)), "inf in resampled ivar")
        self.assertTrue(np.any(out_ivar > 0), "every bin was discarded")

    def test_sn_cons_zero_ivar_leaves_clean_bins_untouched(self):
        """Masking one pixel must not perturb bins that do not overlap it."""
        from ..specutils.resample import _sn_conserving_resample

        clean = np.full(self.n, 4.0)
        holed = clean.copy()
        holed[1000] = 0.0
        _, ivar_clean = _sn_conserving_resample(
            self.wave, self.model.copy(), self.wave_new, ivar=clean.copy()
        )
        _, ivar_holed = _sn_conserving_resample(
            self.wave, self.model.copy(), self.wave_new, ivar=holed
        )
        survived = ivar_holed > 0
        np.testing.assert_array_equal(
            ivar_holed[survived], ivar_clean[survived]
        )

    def test_sn_cons_masked_region_maps_to_zero_ivar(self):
        """A masked wavelength range must come out masked, and only it."""
        from ..specutils.resample import _sn_conserving_resample

        ivar = np.full(self.n, 4.0)
        ivar[(self.wave > 4800.0) & (self.wave < 4900.0)] = 0.0
        _, out_ivar = _sn_conserving_resample(
            self.wave, self.model.copy(), self.wave_new, ivar=ivar
        )
        inside = (self.wave_new > 4810.0) & (self.wave_new < 4890.0)
        outside = (self.wave_new > 4300.0) & (self.wave_new < 4700.0)
        self.assertTrue(np.all(out_ivar[inside] == 0.0))
        self.assertTrue(np.all(out_ivar[outside] > 0.0))

    def test_sn_cons_unmasked_input_is_unaffected(self):
        """With no masked pixels the result must match the original formula.

        Pins backward compatibility: the zero-ivar handling must be inert when
        there is no zero ivar.
        """
        from ..specutils.resample import _project, _sn_conserving_resample

        ivar = np.random.default_rng(1).uniform(1.0, 9.0, self.n)

        # The pre-fix expression, inline.
        flux_per_bin = self.model * np.gradient(self.wave)
        projection = _project(self.wave, self.wave_new)
        expect_flux = projection.dot(flux_per_bin) / np.gradient(self.wave_new)
        scaled = ivar / np.gradient(self.wave) ** 2.0
        expect_ivar = 1.0 / projection.dot(scaled ** (-1.0))
        expect_ivar *= np.gradient(self.wave_new) ** 2.0

        got_flux, got_ivar = _sn_conserving_resample(
            self.wave, self.model.copy(), self.wave_new, ivar=ivar.copy()
        )
        np.testing.assert_array_equal(got_flux, expect_flux)
        np.testing.assert_array_equal(got_ivar, expect_ivar)

    def test_sn_cons_rejects_overhanging_output_grid(self):
        """_project needs the output grid inside the input grid (#48).

        Bin edges extend half a bin beyond the first and last samples, so an
        output grid starting at the input grid's first sample overhangs. That
        used to surface as an opaque numpy error from inside _project.
        """
        from ..specutils.resample import _sn_conserving_resample

        with self.assertRaises(ValueError) as caught:
            _sn_conserving_resample(
                self.wave,
                self.model.copy(),
                np.arange(self.wave[0], self.wave[-1], 1.6),
            )
        self.assertIn("within the input grid", str(caught.exception))


class TestCoaddFlux(unittest.TestCase):
    """Stacking of normalized spectra."""

    def setUp(self):
        self.wave = np.linspace(4000.0, 5000.0, 100)
        self.flux = np.ones((6, 100))
        self.ivar = np.full((6, 100), 4.0)

    def test_identical_spectra_stack_to_themselves(self):
        """Coadding N copies of one spectrum returns that spectrum."""
        for method in ("mean", "ivar-weighted-mean", "irms-weighted-mean"):
            with self.subTest(method=method):
                flux, _ = coadd_flux(
                    self.wave, self.flux.copy(), self.ivar.copy(),
                    method=method, bootstrap=False,
                )
                np.testing.assert_allclose(flux, 1.0)

    def test_mean_recovers_the_mean(self):
        flux_in = np.repeat(np.array([[1.0], [3.0]]), 100, axis=1)
        flux, _ = coadd_flux(
            self.wave, flux_in, np.full((2, 100), 4.0),
            method="mean", bootstrap=False,
        )
        np.testing.assert_allclose(flux, 2.0)

    def test_stacking_reduces_the_uncertainty(self):
        """Stacking N spectra must not leave the ivar worse than one spectrum."""
        _, ivar = coadd_flux(
            self.wave, self.flux.copy(), self.ivar.copy(),
            method="mean", bootstrap=False,
        )
        self.assertTrue(np.all(ivar >= 4.0))

    def test_bootstrap_returns_matching_shape(self):
        flux, ivar = coadd_flux(
            self.wave, self.flux.copy(), self.ivar.copy(),
            method="mean", bootstrap=True, bootstrap_samples=16, n_workers=1,
        )
        self.assertEqual(flux.shape, (100,))
        self.assertEqual(ivar.shape, (100,))

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            coadd_flux(
                self.wave, self.flux.copy(), self.ivar.copy(),
                method="nope", bootstrap=False,
            )

    def test_weighted_quantiles_median_of_uniform_weights(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        got = weighted_quantiles(values, np.ones(5), quantiles=0.5)
        self.assertEqual(got, 3.0)


@unittest.skipUnless(HAVE_DUST, "requires $DUST_DIR for MW dust correction")
class TestStackSpectra(unittest.TestCase):
    """End-to-end stacking. Regression for #42 and #44."""

    def setUp(self):
        from ..specutils.stack import stack_spectra

        self.stack_spectra = stack_spectra
        self.redshift = np.linspace(0.10, 0.15, 4)

    def _run(self, use_mask=False, **kwargs):
        flux, wave, ivar, mask = camera_dicts()
        if use_mask:
            mask["b"][:, :100] = 1
        return self.stack_spectra(
            flux=flux, wave=wave, ivar=ivar,
            mask=(mask if use_mask else None),
            fibermap=fibermap(), redshift=self.redshift,
            bootstrap=False, n_workers=1, **kwargs
        )

    def test_runs_with_default_arguments(self):
        """The documented defaults must not crash (#42).

        norm_flux_window defaults to None and used to be dereferenced
        unconditionally, so median/mean/luminosity were all unreachable.
        """
        (flux, ivar), grid = self._run()
        self.assertEqual(flux.shape, grid.shape)
        self.assertTrue(np.any(np.isfinite(flux)))

    def test_runs_with_a_mask(self):
        """A masked input must not trip the coadd_cameras unpack (#44)."""
        (flux, _), grid = self._run(use_mask=True)
        self.assertEqual(flux.shape, grid.shape)

    def test_mask_removes_signal(self):
        """Masking pixels must leave fewer usable pixels than not masking."""
        (unmasked, _), _ = self._run(use_mask=False)
        (masked, _), _ = self._run(use_mask=True)
        self.assertLess(
            int(np.isfinite(masked).sum()), int(np.isfinite(unmasked).sum())
        )

    def test_each_normalization_method(self):
        for method, extra in (
            ("median", {}),
            ("mean", {}),
            ("flux-window", {"norm_flux_window": (4000.0, 4500.0)}),
        ):
            with self.subTest(norm_method=method):
                (flux, _), grid = self._run(norm_method=method, **extra)
                self.assertEqual(flux.shape, grid.shape)

    def test_flux_window_without_a_window_raises(self):
        """Asking for flux-window without a window is a clear error (#42)."""
        with self.assertRaises(ValueError):
            self._run(norm_method="flux-window")

    def test_missing_redshift_raises(self):
        flux, wave, ivar, _ = camera_dicts()
        with self.assertRaises(ValueError):
            self.stack_spectra(
                flux=flux, wave=wave, ivar=ivar, fibermap=fibermap(),
                redshift=None, bootstrap=False, n_workers=1,
            )


class TestReleaseLayout(unittest.TestCase):
    """Data-release discovery and path resolution in spectra_io.

    Built on a synthetic directory tree that mirrors the real layouts, so
    these need no DESI data and run instantly. The layouts reproduced here
    are, as of 2026-10:

    ==========  ============  ================================
    release     coadd dir     zall-pix location
    ==========  ============  ================================
    fuji        healpix/      zcatalog/
    guadalupe   healpix/      zcatalog/v1/
    iron        healpix/      zcatalog/v1/   (+ v0, + deprecated/)
    jura        none          zcatalog/v1/
    kibo        healpix/      zcatalog/v1.1/ (+ v1)
    loa         healpix/      zcatalog/v1/   (+ deprecated/)
    matterhorn  spectra/      zcatalog/v2/zall/
    ==========  ============  ================================
    """

    def setUp(self):
        import tempfile

        from ..specutils import spectra_io

        self.spectra_io = spectra_io
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

        def make(release, coadd, catalogs):
            base = self.root / release
            if coadd:
                (base / coadd / "main" / "bright" / "80" / "8072").mkdir(parents=True)
            for rel_dir in catalogs:
                d = base / "zcatalog" / rel_dir if rel_dir else base / "zcatalog"
                d.mkdir(parents=True, exist_ok=True)
                (d / f"zall-pix-{release}.fits").touch()

        make("fuji", "healpix", [""])
        make("guadalupe", "healpix", ["v1"])
        make("iron", "healpix", ["v0", "v1", "v1/deprecated"])
        make("jura", None, ["v1"])
        make("kibo", "healpix", ["v1", "v1.1"])
        make("loa", "healpix", ["v1", "v1/deprecated"])
        make("matterhorn", "spectra", ["v2/zall"])
        # A personal reduction: has a zcatalog, but no release-named catalog.
        (self.root / "dylang" / "zcatalog" / "v2").mkdir(parents=True)
        (self.root / "dylang" / "healpix").mkdir(parents=True)

    def test_list_releases_finds_real_releases_only(self):
        """Personal reduction directories must not be reported as releases."""
        found = self.spectra_io.list_releases(
            spectro_redux=self.root, require_spectra=False
        )
        self.assertEqual(
            found,
            ["fuji", "guadalupe", "iron", "jura", "kibo", "loa", "matterhorn"],
        )
        self.assertNotIn("dylang", found)

    def test_list_releases_excludes_catalog_only_by_default(self):
        """jura has a zcatalog but no coadds, so it cannot serve spectra."""
        found = self.spectra_io.list_releases(spectro_redux=self.root)
        self.assertNotIn("jura", found)
        self.assertIn("matterhorn", found)

    def test_coadd_dir_handles_both_names(self):
        """healpix/ up to loa, spectra/ from matterhorn on."""
        self.assertEqual(
            self.spectra_io._coadd_dir(self.root / "loa").name, "healpix"
        )
        self.assertEqual(
            self.spectra_io._coadd_dir(self.root / "matterhorn").name, "spectra"
        )

    def test_coadd_dir_raises_for_catalog_only_release(self):
        """A release with no coadds must say so, not fail obscurely later."""
        with self.assertRaises(FileNotFoundError) as caught:
            self.spectra_io._coadd_dir(self.root / "jura")
        self.assertIn("jura", str(caught.exception))

    def test_zcatalog_path_per_layout(self):
        """Every layout the zcatalog has used must resolve."""
        expected = {
            "fuji": "zcatalog/zall-pix-fuji.fits",
            "guadalupe": "zcatalog/v1/zall-pix-guadalupe.fits",
            "iron": "zcatalog/v1/zall-pix-iron.fits",
            "jura": "zcatalog/v1/zall-pix-jura.fits",
            "kibo": "zcatalog/v1.1/zall-pix-kibo.fits",
            "loa": "zcatalog/v1/zall-pix-loa.fits",
            "matterhorn": "zcatalog/v2/zall/zall-pix-matterhorn.fits",
        }
        for release, tail in expected.items():
            with self.subTest(release=release):
                got = self.spectra_io._zcatalog_path(release, self.root / release)
                self.assertEqual(
                    got.relative_to(self.root / release).as_posix(), tail
                )

    def test_zcatalog_path_prefers_latest_version(self):
        """kibo's v1.1 must win over v1, and iron's v1 over v0."""
        self.assertIn(
            "v1.1",
            str(self.spectra_io._zcatalog_path("kibo", self.root / "kibo")),
        )
        self.assertIn(
            "v1/",
            str(self.spectra_io._zcatalog_path("iron", self.root / "iron")),
        )

    def test_zcatalog_path_ignores_deprecated_copies(self):
        """iron and loa keep a stale copy under v1/deprecated/."""
        for release in ("iron", "loa"):
            with self.subTest(release=release):
                got = self.spectra_io._zcatalog_path(release, self.root / release)
                self.assertNotIn("deprecated", str(got))

    def test_zcatalog_path_missing_release_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.spectra_io._zcatalog_path("nickel", self.root / "nickel")

    def test_version_ordering(self):
        """v1.1 sorts above v1, and v10 above v9."""
        key = self.spectra_io._version_sort_key
        self.assertGreater(key("v1.1"), key("v1"))
        self.assertGreater(key("v10"), key("v9"))
        self.assertGreater(key("v2"), key("v1.9"))

    def test_missing_targets_raise_valueerror_naming_them(self):
        """Targets absent from the catalog must be reported, not KeyError'd.

        The check used to sit *after* ``sel_data.loc[targetids]``, so pandas
        raised ``KeyError: '[...] not in index'`` first and the intended
        message was unreachable. It was also missing its f-string prefix, so
        it would have printed the literal braces had it ever run (#32).
        """
        import unittest.mock as mock

        import pandas as pd

        # Catalog knows target 2 only; 1 and 3 are missing.
        frame = pd.DataFrame(
            {
                "SURVEY": ["main"],
                "PROGRAM": ["bright"],
                "HEALPIX": [8072],
                "TARGETID": [2],
            }
        )
        with mock.patch.dict(
            os.environ, {"DESI_SPECTRO_REDUX": str(self.root)}
        ), mock.patch.object(
            self.spectra_io, "_sel_objects_fits", return_value=frame
        ):
            with self.assertRaises(ValueError) as caught:
                self.spectra_io.get_spectra(
                    [1, 2, 3], "fuji", n_workers=1, use_db=False
                )
        message = str(caught.exception)
        self.assertIn("1", message)
        self.assertIn("3", message)
        self.assertIn("fuji", message)
        self.assertNotIn("{", message, "message is not an f-string")

    def test_healpix_column_accepts_both_names(self):
        """matterhorn renamed HEALPIX to UNIQPIX."""
        pick = self.spectra_io._healpix_column
        self.assertEqual(pick(["TARGETID", "HEALPIX"]), "HEALPIX")
        self.assertEqual(pick(["TARGETID", "UNIQPIX"]), "UNIQPIX")
        with self.assertRaises(KeyError):
            pick(["TARGETID", "SURVEY"])


class TestKnownBugs(unittest.TestCase):
    """Defects that are still open, recorded as expected failures.

    These document the P2 findings from the code review. An unexpected success
    here means the underlying bug has been fixed and the test should be moved
    into the class it belongs to.
    """

    @unittest.expectedFailure
    def test_coadd_cameras_returns_the_mask(self):
        """coadd_cameras returns None instead of a mask (issue #4).

        The guard reads ``if mask is not None`` against a local that is never
        assigned, instead of ``if mask_cam is not None``.
        """
        flux, wave, ivar, mask = camera_dicts(nspec=2)
        _, _, _, out_mask = coadd_cameras(flux, wave, ivar, mask)
        self.assertIsNotNone(out_mask)

    @unittest.expectedFailure
    def test_resample_honours_fill_val(self):
        """resample() hardcodes fill_val=np.nan in every dispatch branch.

        The user's fill_val argument is accepted and then discarded.
        """
        wave = np.arange(4000.0, 6000.0, 0.8)
        nspec, n = 2, len(wave)
        flux = np.ones((nspec, n))
        ivar = np.full((nspec, n), 4.0)
        # An output grid wider than the input, so some bins are unreachable.
        out_flux, _ = resample(
            np.arange(3000.0, 7000.0, 1.6), np.tile(wave, (nspec, 1)),
            flux, ivar, method="linear", fill_val=-999.0, n_workers=1,
        )
        self.assertTrue(np.any(out_flux == -999.0))

    @unittest.expectedFailure
    def test_coadd_flux_does_not_mutate_its_input(self):
        """_coadd_flux writes zeros and 1e-10 into the caller's arrays."""
        flux = np.ones((4, 20))
        ivar = np.full((4, 20), 4.0)
        ivar[0, :5] = 0.0  # make some pixels masked so the nan_mask bites
        flux_before, ivar_before = flux.copy(), ivar.copy()
        coadd_flux(np.linspace(4000.0, 4100.0, 20), flux, ivar,
                   method="mean", bootstrap=False)
        np.testing.assert_array_equal(flux, flux_before)
        np.testing.assert_array_equal(ivar, ivar_before)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
