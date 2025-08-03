use crate::glcm::map_glcm;
use crate::ui::MapOpts;
use array_lib::ArrayDim;
use num_traits::{ToPrimitive, Zero};
use rayon::prelude::*;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
pub mod glcm;
mod subr;
pub mod ui;

pub fn run_glcm_map(
    opts: MapOpts,
    image: Vec<f64>,
    mask: Option<Vec<f64>>,
    dims: ArrayDim,
    progress: Arc<AtomicUsize>,
) -> (Vec<f32>, ArrayDim) {
    let vol_dims = &dims.shape()[0..3];

    let mut bins = vec![0; dims.numel()];

    let mask = if let Some(mask) = mask {
        mask
    } else {
        vec![1.; dims.numel()]
    };

    let mask: Vec<u16> = mask
        .into_iter()
        .map(|x| if x > 0. { 1u16 } else { 0u16 })
        .collect();

    discretize_by_bin_count(opts.n_bins, &image, &mask, &mut bins);

    let mut angles = vec![[0, 0, 0]; n_angles(opts.kernel_radius)];
    generate_angles(&mut angles, opts.kernel_radius);

    let odims = ArrayDim::from_shape(&[24, vol_dims[0], vol_dims[1], vol_dims[2]]);

    let mut out = vec![0f64; odims.numel()];

    map_glcm(
        &opts.features(),
        vol_dims,
        &bins,
        &mut out,
        &angles,
        opts.n_bins,
        opts.kernel_radius,
        &[],
        opts.max_threads,
        progress,
    );

    let builder = rayon::ThreadPoolBuilder::new();
    let thread_pool = if let Some(max_threads) = opts.max_threads {
        builder.num_threads(max_threads).build().unwrap()
    } else {
        builder.build().unwrap()
    };
    thread_pool.install(|| {
        // save some memory
        let out: Vec<_> = out.into_par_iter().map(|x| x as f32).collect();
        let mut features = odims.alloc(0f32);
        change_dims(vol_dims, 24, &out, &mut features);
        let fdims = ArrayDim::from_shape(&[vol_dims[0], vol_dims[1], vol_dims[2], 24]);
        (features, fdims)
    })
}

/// discretize gray level intensities 'x' into 'n_bins' within some roi 'mask'. The result is
/// written to 'd' as unsigned 16-bit values
pub fn discretize_by_bin_count<M: Zero + Copy + Send + Sync>(
    n_bins: usize,
    x: &[f64],
    mask: &[M],
    d: &mut [u16],
) {
    assert!(n_bins > 0);
    assert!(n_bins < u16::MAX as usize);
    assert_eq!(mask.len(), x.len());
    assert_eq!(d.len(), x.len());

    // determine the range of the input intensities in the masked region
    let g = x
        .par_iter()
        .zip(mask.par_iter())
        .filter(|(_, m)| !m.is_zero())
        .map(|(x, _)| *x)
        .collect::<Vec<f64>>();
    let max = *g
        .iter()
        .max_by(|a, b| a.partial_cmp(b).expect("unable to compare values"))
        .expect("no max value found");
    let min = *g
        .iter()
        .min_by(|a, b| a.partial_cmp(b).expect("unable to compare values"))
        .expect("no min value found");

    assert!(max.is_finite());
    assert!(min.is_finite());

    let range = max - min;

    // discretize x into bins
    d.par_iter_mut()
        .zip(mask.par_iter().zip(x.par_iter()))
        .for_each(|(d, (m, &x))| {
            if m.is_zero() {
                *d = 0;
                return;
            }
            *d = if x < max {
                ((n_bins as f64 * (x - min) / range).floor() + 1.) as u16
            } else {
                n_bins as u16
            };
        })
}

/// discretize gray level intensities 'x' into bins within some roi 'mask' given a 'bin_width.' The result is
/// written to 'd' as unsigned 16-bit values
pub fn discretize_by_bin_width<M: Zero + Copy + Send + Sync>(
    bin_width: f64,
    x: &[f64],
    mask: &[M],
    d: &mut [u16],
) {
    assert_eq!(mask.len(), x.len());
    assert_eq!(d.len(), x.len());

    let min = x
        .par_iter()
        .zip(mask.par_iter())
        .filter(|(_, m)| !m.is_zero())
        .map(|(x, _)| *x)
        .min_by(|a, b| a.partial_cmp(b).expect("unable to compare values"))
        .expect("no min value found");

    assert!(min.is_finite());

    // discretize x into bins
    d.iter_mut()
        .zip(mask.iter().zip(x.iter()))
        .for_each(|(d, (m, &x))| {
            if m.is_zero() {
                *d = 0;
                return;
            }
            let disc: f64 = (x / bin_width).floor() - (min / bin_width).floor() + 1.;
            *d = disc
                .to_u16()
                .expect("number of bins exceeds u16::MAX with this bin width");
        })
}

/// returns the number of angles associated with some infinity norm 'r' in 3-D
pub fn n_angles(r: usize) -> usize {
    ((r * 2 + 1).pow(3) - 1) / 2
}

/// generate a set of angles with some distance 'r' (infinity norm)
pub fn generate_angles(angles: &mut [[i32; 3]], r: usize) {
    assert!(r >= 1, "r must be positive");
    let n_expected = n_angles(r);
    assert_eq!(
        angles.len(),
        n_expected,
        "buffer must have exactly {n_expected} elements"
    );

    let mut idx = 0usize;

    let r = r as i32;
    for x in (0..=r).rev() {
        for y in (-r..=r).rev() {
            for z in (-r..=r).rev() {
                if x == 0 && y == 0 && z == 0 {
                    continue; // skip the origin
                }

                // Take only the vector whose first non‑zero entry is positive.
                let first_non_zero = if x != 0 {
                    x
                } else if y != 0 {
                    y
                } else {
                    z
                };
                if first_non_zero < 0 {
                    continue; // this is the additive inverse we drop
                }

                angles[idx] = [x, y, z];
                idx += 1;
            }
        }
    }

    debug_assert_eq!(idx, angles.len());
}

/// swaps the feature dims to the last dimension
pub fn change_dims<T: Sized + Copy + Send + Sync>(
    dims: &[usize],
    n_features: usize,
    x: &[T],
    y: &mut [T],
) {
    let i_dims = ArrayDim::from_shape(&[n_features, dims[0], dims[1], dims[2]]);
    let o_dims = ArrayDim::from_shape(&[dims[0], dims[1], dims[2], n_features]);
    y.par_iter_mut().enumerate().for_each(|(i, y)| {
        let [i, j, k, f, ..] = o_dims.calc_idx(i);
        let addr = i_dims.calc_addr(&[f, i, j, k]);
        *y = x[addr];
    });
}

#[cfg(test)]
mod tests {
    use crate::glcm::map_glcm;
    use crate::ui::MapOpts;
    use crate::{
        change_dims, discretize_by_bin_count, discretize_by_bin_width, generate_angles, n_angles,
    };
    use array_lib::io_nifti::{read_nifti, write_nifti};
    use array_lib::ArrayDim;
    use rand::Rng;
    use rayon::prelude::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    #[test]
    fn test_regression() {
        let opts = test_regresion_params();

        let (img, idims, ..) = read_nifti::<u16>("regression_test_img.nii");
        let (expected_result, odims, ..) = read_nifti::<f64>("regression_test_result.nii");

        let vol_dims = &idims.shape()[0..3];

        let mut angles = vec![[0, 0, 0]; n_angles(opts.kernel_radius)];
        generate_angles(&mut angles, opts.kernel_radius);
        let mut out = vec![0f64; odims.numel()];
        let mut swapped = vec![0f64; odims.numel()];
        let prog = Arc::new(AtomicUsize::new(0));
        map_glcm(
            &opts.features(),
            vol_dims,
            &img,
            &mut out,
            &angles,
            opts.n_bins,
            opts.kernel_radius,
            &[],
            None,
            prog.clone(),
        );

        change_dims(vol_dims, 24, &out, &mut swapped);

        let err_norm = expected_result
            .iter()
            .zip(&swapped)
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();
        // very small residual norm
        assert!(err_norm < 1e-15);
    }

    fn test_regresion_params() -> MapOpts {
        let n_bins = 64;
        let kernel_radius = 1;
        let mut opts = MapOpts::default();
        opts.kernel_radius = kernel_radius;
        opts.n_bins = n_bins;
        opts
    }

    /// test to ensure the mapping output is stable across code updates
    fn regression_setup() {
        let opts = test_regresion_params();

        let mut angles = vec![[0, 0, 0]; n_angles(opts.kernel_radius)];
        generate_angles(&mut angles, opts.kernel_radius);

        let vol_dims = [10, 10, 10];
        let dims = ArrayDim::from_shape(&vol_dims);
        let o_dims = ArrayDim::from_shape(&[vol_dims[0], vol_dims[1], vol_dims[2], 24]);
        let mut x = vec![0u16; dims.numel()];
        let mut rng = rand::rng();
        x.iter_mut()
            .for_each(|x| *x = rng.random_range(1..=opts.n_bins as u16));

        let mut m = vec![0u8; dims.numel()];
        m.iter_mut().for_each(|m| *m = rng.random_range(0..=1));

        let img: Vec<u16> = x
            .into_iter()
            .zip(m)
            .map(|(x, m)| if m > 0 { x } else { 0 })
            .collect();
        let mut out = vec![0f64; dims.numel() * 24];

        // pre-masked
        write_nifti("regression_test_img", &img, dims);
        let prog = Arc::new(AtomicUsize::new(0));
        map_glcm(
            &opts.features(),
            &vol_dims,
            &img,
            &mut out,
            &angles,
            opts.n_bins,
            opts.kernel_radius,
            &[],
            None,
            prog,
        );

        let mut swapped = vec![0f64; out.len()];
        change_dims(&vol_dims, 24, &out, &mut swapped);
        write_nifti("regression_test_result", &swapped, o_dims);
    }

    #[test]
    /// tests result against pyradiomics output as generated by pyrad_test.py
    fn test_pyrad_comparison() {
        let dims = [3, 3, 3];
        let patch_radius = 1;
        let n_bins: usize = 5;
        let n_features = 24;
        let x_dims = ArrayDim::from_shape(&dims);
        let f_dims = ArrayDim::from_shape(&[n_features, dims[0], dims[1], dims[2]]);
        let mut x = x_dims.alloc(0u16);
        let mut features = f_dims.alloc(0f64);

        // vox to compute (center voxel of 3x3x3)
        let vox = [1, 1, 1];

        // angles must be on the half-shell, meaning that no additive inverses exist
        let angles = [
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, -1],
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, -1],
            [1, -1, 1],
            [1, -1, 0],
            [1, -1, -1],
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, -1],
            [0, 0, 1],
        ];

        let n_angles = angles.len();
        x.copy_from_slice(&[
            0, 1, 2, 0, 4, 5, 3, 4, 2, 3, 2, 1, 4, 2, 5, 3, 1, 1, 5, 3, 2, 1, 1, 5, 4, 4, 0,
        ]);
        let x = x;

        let opts = MapOpts::default();
        let prog = Arc::new(AtomicUsize::new(0));
        map_glcm(
            &opts.features(),
            &dims,
            &x,
            &mut features,
            &angles,
            n_bins,
            patch_radius,
            &[vox],
            None,
            prog.clone(),
        );

        let mut out = vec![];
        for i in 0..n_features {
            let f_addr = f_dims.calc_addr(&[i, vox[0] as usize, vox[1] as usize, vox[2] as usize]);
            println!("{}: {:?}", i + 1, features[f_addr]);
            out.push(features[f_addr]);
        }

        let pyrad_features: [f64; 24] = [
            7.411452436452436,
            2.791798479298479,
            29.013681805845312,
            0.2509018267792349,
            3.4377282528878843,
            5.024200799200798,
            -0.1699681782433016,
            1.8279248529248524,
            2.0075007066521584,
            1.4871883915146902,
            0.08812285938368189,
            3.6403624589991965,
            -0.3800144142470373,
            0.8867907665221482,
            0.3889951714951715,
            0.7156581005871134,
            0.8554885844258353,
            0.46506627631627634,
            0.7597205318931509,
            0.3653838252796586,
            0.13202630702630705,
            5.583596958596958,
            2.3932244789829733,
            2.1154822630221704,
        ];

        /*
            If a feature class has a function _calculateCMatrix(), identifying it as a C enhanced class,
            output from the C extension is compared to the output from full python calculation.
            An absolute difference of 1e-3 is allowed to account for machine precision errors.
        */

        // We will be more strict that 1e-3 here
        let error: Vec<_> = out
            .iter()
            .zip(&pyrad_features)
            .map(|(&y, &x)| x - y)
            .collect();
        let norm_err = error.iter().map(|e| e.powi(2)).sum::<f64>().sqrt();
        println!("norm_err: {norm_err:?}");
        assert!(error.iter().all(|&e| e < 1.0e-14));
    }

    #[test]
    fn test_discretize() {
        let x = [1., 3., 10., 11., 15., 62.];
        let m = [0, 1, 1, 1, 1, 1];
        let mut d = vec![0; x.len()];
        discretize_by_bin_count(3, &x, &m, &mut d);
        assert_eq!(d, [0, 1, 1, 1, 1, 3]);
        discretize_by_bin_width(2., &x, &m, &mut d);
        assert_eq!(d, [0, 1, 5, 5, 7, 31]);
    }

    #[test]
    fn test_angles() {
        let r = 1;
        let n = n_angles(r);
        let mut angles = vec![[0, 0, 0]; n];
        generate_angles(&mut angles, r);
        let expected = [
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, -1],
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, -1],
            [1, -1, 1],
            [1, -1, 0],
            [1, -1, -1],
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, -1],
            [0, 0, 1],
        ];
        assert_eq!(angles, expected);
    }
}
