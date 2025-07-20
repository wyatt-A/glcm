use ndarray::ShapeBuilder;
use ndarray_linalg::EigVals;

#[inline]
pub fn calc_sum_of_squares(glcm:&[f64],n_bins:usize,ux:f64) -> f64 {
    let mut s = 0.;
    for i in 0..n_bins {
        for j in 0..n_bins {
            s += glcm[j*n_bins + i] * ((i + 1) as f64 - ux).powi(2)
        }
    }
    s
}

#[inline]
pub fn calc_sum_entropy(pxpy:&[f64],eps:f64) -> f64 {
    - pxpy.iter().filter(|x| **x > 0.).map(|&pxpy| {
        pxpy * (pxpy + eps).log2()
    }).sum::<f64>()
}

#[inline]
pub fn calc_sum_average(pxpy:&[f64],k_vals_sum:&[f64]) -> f64 {
    debug_assert_eq!(k_vals_sum.len(),pxpy.len());
    k_vals_sum.iter().zip(pxpy.iter()).map(|(&k,&pxpy)| {
        k * pxpy
    }).sum()
}

#[inline]
pub fn calc_max_prob(glcm:&[f64]) -> f64 {
    *glcm.iter().max_by(|a,b| a.partial_cmp(b)
        .expect("number must be finite"))
        .unwrap_or(&0.)
}

#[inline]
pub fn calc_inverse_var(pxmy:&[f64],k_vals_diff:&[f64]) -> f64 {
    debug_assert_eq!(k_vals_diff.len(),pxmy.len());
    k_vals_diff.iter().zip(pxmy.iter()).filter(|(k,_)|**k > 0.).map(|(&k,&pxmy)| {
        pxmy / k.powi(2)
    }).sum()
}

#[inline]
pub fn calc_idn(pxmy:&[f64],k_vals_diff:&[f64],n_bins:usize) -> f64 {
    debug_assert_eq!(k_vals_diff.len(),pxmy.len());
    k_vals_diff.iter().zip(pxmy.iter()).map(|(&k,&pxmy)| {
        pxmy / (1. + (k/n_bins as f64))
    }).sum()
}

#[inline]
pub fn calc_id(pxmy:&[f64],k_vals_diff:&[f64]) -> f64 {
    debug_assert_eq!(k_vals_diff.len(),pxmy.len());
    k_vals_diff.iter().zip(pxmy.iter()).map(|(&k,&pxmy)| {
        pxmy / (1. + k)
    }).sum()
}

#[inline]
pub fn calc_idmn(pxmy:&[f64],k_vals_diff:&[f64],n_bins:usize) -> f64 {
    debug_assert_eq!(k_vals_diff.len(),pxmy.len());
    k_vals_diff.iter().zip(pxmy.iter()).map(|(&k,&pxmy)| {
        pxmy / (1. + (k.powi(2)/(n_bins.pow(2) as f64)))
    }).sum()
}


#[inline]
pub fn calc_mcc(glcm:&[f64], scratch_matrix:&mut [f64], n_bins:usize, px:&[f64], py:&[f64], eps:f64) -> f64 {
    assert_eq!(glcm.len(), scratch_matrix.len());
    assert_eq!(glcm.len(), n_bins*n_bins);
    scratch_matrix.fill(0.0);

    // build the Q matrix
    for i in 0..n_bins {
        for j in 0..n_bins {
            let mut s = 0.;
            for k in 0..n_bins {
                s += glcm[k*n_bins + i] * glcm[k*n_bins + j] / (px[i]*py[k] + eps);
            }
            scratch_matrix[j*n_bins + i] = s;
        }
    }

    // calculate the sqrt of the second-largest eigenvalue
    let m = ndarray::ArrayView2::from_shape((n_bins,n_bins).f(),&scratch_matrix).unwrap();
    let mut e = m.eigvals().unwrap().to_vec();

    if e.len() < 2 {
        1.
    }else {
        e.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
        e[n_bins - 2].re.sqrt()
    }

}




#[inline]
pub fn calc_idm(pxmy:&[f64],k_vals_diff:&[f64]) -> f64 {
    debug_assert_eq!(k_vals_diff.len(),pxmy.len());
    k_vals_diff.iter().zip(pxmy.iter()).map(|(&k,&pxmy)| {
        pxmy / (1. + k.powi(2))
    }).sum()
}

// #[inline]
// pub fn calc_imc(glcm:&[f64],n_bins:usize,px:&[f64],py:&[f64],eps:f64) -> [f64;2] {
//
//     let hx = joint_entropy(px,eps);
//     let hy = joint_entropy(py,eps);
//     let hxy = joint_entropy(glcm,eps);
//
//     let hxy1 = calc_hxy1(glcm,n_bins,px,py,eps);
//     let hxy2 = calc_hxy2(n_bins,px,py,eps);
//
//     let imc1 = (hxy - hxy1) / hx.max(hy);
//     let a = 1. - (- 2. * (hxy2 - hxy)).exp();
//     let imc2 = if a < 0. {
//         0.
//     }else {
//         a.sqrt()
//     };
//     [imc1,imc2]
// }

pub fn calc_imc1(
    glcm: &[f64],
    n_bins: usize,
    px: &[f64],
    py: &[f64],
    hxy: f64,
    eps: f64,
) -> f64 {

    debug_assert_eq!(py.len(), n_bins);
    debug_assert_eq!(glcm.len(), n_bins * n_bins);

    // HX = -Σ px * log2(px + eps)
    let mut hx = 0.0;
    for &p in px {
        hx -= p * (p + eps).log2();
    }

    // HY = -Σ py * log2(py + eps)
    let mut hy = 0.0;
    for &p in py {
        hy -= p * (p + eps).log2();
    }

    // HXY1 = -Σ_ij p(i,j) * log2(px(i) * py(j) + eps)
    let mut hxy1 = 0.0;
    for i in 0..n_bins {
        let px_i = px[i];
        for j in 0..n_bins {
            let py_j = py[j];
            let pij = glcm[j * n_bins + i]; // column-major
            hxy1 -= pij * (px_i * py_j + eps).log2();
        }
    }

    let num = hxy - hxy1;          // Note: this is -(mutual information)
    let denom = hx.max(hy);

    if denom == 0.0 {
        0.0                         // flat region convention
    } else {
        num / denom
    }
}

#[inline]
pub fn calc_imc2(
    px: &[f64],
    py: &[f64],
    hxy: f64,
    eps: f64,
) -> f64 {
    let n_bins = px.len();
    assert_eq!(py.len(), n_bins);

    // HXY2 = -Σ_i Σ_j  (px_i * py_j) * log₂(px_i * py_j + eps)
    let mut hxy2 = 0.0;
    for &px_i in px {
        for &py_j in py {
            let prod = px_i * py_j;
            hxy2 -= prod * (prod + eps).log2();
        }
    }

    // Numerical guard: if HXY ≥ HXY2 (shouldn’t happen, but may due to rounding),
    // PyRadiomics returns 0.
    if hxy >= hxy2 {
        return 0.0;
    }

    // IMC2 = √[ 1 − exp(‑2·(HXY2 − HXY)) ]
    let exponent = (-2.0 * (hxy2 - hxy)).exp();   // natural exponential
    let val = 1.0 - exponent;

    // val is ≥ 0 by construction here.  In extremely rare cases it might dip
    // slightly negative due to machine precision; clamp at 0 for safety.
    val.max(0.0).sqrt()
}

#[inline]
fn calc_hxy2(n_bins:usize,px:&[f64],py:&[f64],eps:f64) -> f64 {
    let mut h = 0.;
    for i in 0..n_bins {
        for j in 0..n_bins {
            h += px[i] * py[j] * (px[i] * py[j] + eps ).log2();
        }
    }
    h
}

#[inline]
fn calc_hxy1(glcm:&[f64],n_bins:usize,px:&[f64],py:&[f64],eps:f64) -> f64 {
    let mut h = 0.;
    for i in 0..n_bins {
        for j in 0..n_bins {
            h += glcm[j*n_bins + i] * (px[i] * py[j] + eps ).log2();
        }
    }
    h
}

//HXY=−∑Ngi=1∑Ngj=1p(i,j)log2(p(i,j)+ϵ) be the entropy of p(i,j)

#[inline]
pub fn joint_entropy(x:&[f64], eps:f64) -> f64 {
     - x.iter().filter(|x| **x > 0.).map(|x| *x * (x + eps).log2()).sum::<f64>()
}

#[inline]
pub fn joint_energy(glcm:&[f64]) -> f64 {
    glcm.iter().map(|x| x.powi(2)).sum()
}

#[inline]
pub fn difference_variance(pxmy:&[f64],k_vals_diff:&[f64]) -> f64 {
    let da = difference_average(pxmy,k_vals_diff);
    k_vals_diff.iter().zip(pxmy.iter()).map(|(&k,&pxmy)| {
        (k - da).powi(2) * pxmy
    }).sum()
}

#[inline]
pub fn difference_entropy(pxmy:&[f64],eps:f64) -> f64 {
     - pxmy.iter().filter(|x| **x > 0.).map(|&x|{
            (x + eps).log2() * x
    }).sum::<f64>()
}

#[inline]
pub fn difference_average(pxmy:&[f64],k_vals_diff:&[f64]) -> f64 {
    debug_assert_eq!(k_vals_diff.len(),pxmy.len());
    k_vals_diff.iter().zip(pxmy.iter()).map(|(&k,&v)| v * k).sum()
}

#[inline]
pub fn correlation(glcm: &[f64], n_bins: usize, ux: f64, uy: f64, eps: f64) -> f64 {
    assert_eq!(glcm.len(), n_bins * n_bins);

    let mut corm = 0.0;
    let mut varx = 0.0;
    let mut vary = 0.0;

    for i in 0..n_bins {
        let dx = (i+1) as f64 - ux;
        for j in 0..n_bins {
            let p = glcm[j * n_bins + i];
            let dy = (j+1) as f64 - uy;
            corm += p * dx * dy;
            varx += p * dx * dx;
            vary += p * dy * dy;
        }
    }

    let sigx = varx.sqrt();
    let sigy = vary.sqrt();

    if sigx == 0.0 || sigy == 0.0 {
        1.0
    } else {
        corm / (sigx * sigy + eps)
    }
}

/// returns cluster prominence, shade, tendency, and contrast, in order
pub fn cluster(glcm: &[f64],n_bins:usize, ux:f64, uy:f64) -> [f64;4] {

    let mut prom = 0.;
    let mut shade = 0.;
    let mut tendency = 0.;
    let mut contrast = 0.;

    for i in 0..n_bins {
        for j in 0..n_bins {

            let c1 = (i+1) as f64 + (j+1) as f64;
            let c3 = (i+1) as f64 - (j+1) as f64;
            let c2 = (c1 - ux - uy);

            let c_prom = c2.powi(4);
            let c_shade = c2.powi(3);
            let c_tend = c2.powi(2);
            let c_con = c3.powi(2);

            prom += glcm[j*n_bins + i] * c_prom;
            shade += glcm[j*n_bins + i] * c_shade;
            tendency += glcm[j*n_bins + i] * c_tend;
            contrast += glcm[j*n_bins + i] * c_con;

        }
    }
    [prom, shade, tendency, contrast]
}

pub fn std_dev(x:&[f64]) -> f64 {
    variance(x).sqrt()
}

pub fn variance(x:&[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let variance = x.iter()
        .map(|x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>() / (n as f64 - 1.0);
    variance.sqrt()
}

pub fn auto_correlation(glcm: &[f64], n_bins:usize) -> f64 {
    let mut a = 0f64;
    for i in 0..n_bins {
        for j in 0..n_bins {
            a += glcm[j*n_bins + i] * ((i+1) * (j+1)) as f64;
        }
    }
    a
}


pub fn calc_joint_entropy(glcm: &[f64], epsilon: f64) -> f64 {
    glcm.iter()
        .filter(|&&val| val > 0.0)
        .map(|&val| -val * (val + epsilon).log2())
        .sum()
}

pub fn calc_marginals_inplace(
    glcm: &[f64],
    n_bins: usize,
    px_plus_y: &mut [f64],
    px_minus_y: &mut [f64],
) {
    assert_eq!(glcm.len(), n_bins * n_bins);
    assert_eq!(px_plus_y.len(), 2 * n_bins - 1);
    assert_eq!(px_minus_y.len(), n_bins);

    // Clear the output buffers
    px_plus_y.fill(0.0);
    px_minus_y.fill(0.0);

    for j in 0..n_bins {
        for i in 0..n_bins {
            let idx = j * n_bins + i; // column-major: col * nrow + row
            let val = glcm[idx];

            let sum = i + j;                   // sum range: 0..=2(Ng-1)
            let diff = (i as isize - j as isize).abs() as usize;

            px_plus_y[sum] += val;            // px+y[k], k = i+j in 0..2Ng-2
            px_minus_y[diff] += val;          // px−y[k], k = |i-j| in 0..Ng-1

        }
    }
}

pub fn calc_mean_gray_intensity(p:&[f64], n_bins:usize, intensities:&mut [f64]) {

    let n_angles = p.len() / n_bins;
    debug_assert_eq!(intensities.len(), n_angles);

    p.chunks_exact(n_bins).zip(intensities.iter_mut()).for_each(|(p,intensity)|{
        *intensity = p.iter().enumerate().map(|(i,p)| (i+1) as f64 * p).sum()
    });

}

pub fn calc_marginal_row_prob(glcm:&[f64],n:usize,row_prob:&mut [f64]) {

    debug_assert_eq!(row_prob.len(),n);
    debug_assert_eq!(glcm.len(),n*n);

    row_prob.fill(0.);

    let stride = n;
    row_prob.iter_mut().enumerate().for_each(|(i,p)| {
        for j in 0..n {
            *p += glcm[j*stride + i];
        }
    });

}

pub fn calc_marginal_col_prob(glcm:&[f64],n:usize,col_prob:&mut [f64]) {

    debug_assert_eq!(col_prob.len(),n);
    debug_assert_eq!(glcm.len(),n*n);

    col_prob.fill(0.);

    col_prob.iter_mut().zip(glcm.chunks_exact(n)).for_each(|(p,col)| {
        *p = col.iter().sum();
    });

}






/// return the range given some n and a radius
pub fn get_range_inclusive(patch_radius:usize,r:i32) -> [i32;2] {

    if r == 0 {
        [-(patch_radius as i32),patch_radius as i32]
    }else if r < 0 {
        [-(patch_radius as i32 + r),patch_radius as i32]
    }else {
        [-(patch_radius as i32),(patch_radius as i32) - r]
    }

}



pub fn in_bounds(grid_size:&[usize],test_point:&[i32]) -> Option<[usize;3]> {

    for (&s,&i) in grid_size.iter().zip(test_point.iter()) {
        if i < 0 || i >= s as i32 {
            return None
        }
    }

    Some([
        test_point[0] as usize,
        test_point[1] as usize,
        test_point[2] as usize,
    ])

}

#[inline]
pub fn symmetrize_in_place_f64(a: &mut [f64], n: usize) {
    debug_assert_eq!(a.len(), n * n, "slice length must be n*n");

    for i in 0..n {
        for j in (i + 1)..n {
            let idx_ij = i * n + j;
            let idx_ji = j * n + i;
            let s = a[idx_ij] + a[idx_ji];

            a[idx_ij] = s;
            a[idx_ji] = s;
        }

        let idx_ii = i * n + i;
        a[idx_ii] = a[idx_ii] * 2.;
    }
}

#[inline]
pub fn symmetrize_in_place(a: &mut [usize], n: usize) {
    debug_assert_eq!(a.len(), n * n, "slice length must be n*n");

    for i in 0..n {
        for j in (i + 1)..n {
            let idx_ij = i * n + j;
            let idx_ji = j * n + i;
            let s = a[idx_ij] + a[idx_ji];

            a[idx_ij] = s;
            a[idx_ji] = s;
        }

        let idx_ii = i * n + i;
        a[idx_ii] = a[idx_ii] * 2;
    }
}