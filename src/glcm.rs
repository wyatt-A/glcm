use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use array_lib::ArrayDim;
use crate::subr::{calc_auto_correlation, calc_id, calc_idm, calc_idmn, calc_idn, calc_imc1, calc_imc2, calc_inverse_var, calc_marginal_col_prob, calc_marginal_row_prob, calc_marginals_inplace, calc_max_prob, calc_mcc, calc_mean_gray_intensity, calc_sum_average, calc_sum_entropy, calc_sum_of_squares, calc_cluster, calc_correlation, calc_difference_average, calc_difference_entropy, calc_difference_variance, calc_range_inclusive, in_volume, calc_joint_energy, calc_joint_entropy, std_dev, symmetrize_in_place_f64, in_box};
use rayon::prelude::*;
use strum::{Display, EnumIter};
use crate::ui::MapOpts;

// this allocates a dynamically sized array to store GLCM data (1 per thread)
// once a thread wants to write data to it, it will resize it appropriately
thread_local! {
    static THREAD_GLCM: RefCell<Vec<f64>> = RefCell::new(vec![]);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::time::Instant;
    use array_lib::ArrayDim;
    use array_lib::io_nifti::write_nifti;
    use rand;
    use rand::Rng;
    use crate::change_dims;
    use super::map_glcm;
    use crate::ui::MapOpts;

    #[test]
    fn test_map_speed() {

        let n = 50;
        let n_features = 24;
        let dims = [n,n,n];
        let o_dims = [n_features,n,n,n];
        let patch_radius = 2;
        let n_bins = 32;
        let mut x = ArrayDim::from_shape(&dims).alloc(0u16);
        let mut rng = rand::rng();
        x.iter_mut().for_each(|x| *x = rng.random_range(1..=n_bins as u16));
        let x = x;

        let mut features = ArrayDim::from_shape(&o_dims).alloc(0f64);

        // angles associated with first half-shell
        let angles = [
            [ 1 , 1 , 1],
            [ 1 , 1 , 0],
            [ 1 , 1 ,-1],
            [ 1 , 0 , 1],
            [ 1 , 0 , 0],
            [ 1,  0 ,-1],
            [ 1, -1 , 1],
            [ 1, -1 , 0],
            [ 1 ,-1 ,-1],
            [ 0 , 1 , 1],
            [ 0 , 1 , 0],
            [ 0 , 1 ,-1],
            [ 0,  0 , 1],
        ];

        let opts = MapOpts::default();
        let now = Instant::now();
        let prog = Arc::new(AtomicUsize::new(0));
        map_glcm(&opts.features(),&dims, &x, &mut features, &angles, n_bins, patch_radius,&[],None,prog.clone());
        let dur = now.elapsed();

        let o_dim = ArrayDim::from_shape(&[n,n,n,n_features]);
        let mut out = o_dim.alloc(0f64);

        println!("swapping dims ...");
        change_dims(&dims,n_features,&features,&mut out);

        println!("Elapsed: {:.2?} sec", dur.as_secs_f64());

        let vox_per_sec = x.len() as f64 / dur.as_secs_f64();
        println!("computed {vox_per_sec} voxels / sec")
    }

}

#[derive(Debug,Copy,Clone,Hash,Eq,PartialEq,EnumIter,Display)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum GLCMFeature {
    AUTO_CORRELATION,
    JOINT_AVERAGE,
    CLUSTER_PROMINENCE,
    CLUSTER_SHADE,
    CLUSTER_TENDENCY,
    CONTRAST,
    CORRELATION,
    DIFFERENCE_AVERAGE,
    DIFFERENCE_ENTROPY,
    DIFFERENCE_VARIANCE,
    JOINT_ENERGY,
    JOINT_ENTROPY,
    IMC1,
    IMC2,
    INVERSE_DIFFERENCE_MOMENT,
    MAXIMUM_CORRELATION_COEFFICIENT,
    INVERSE_DIFFERENCE_MOMENT_NORMALIZED,
    INVERSE_DIFFERENCE,
    INVERSE_DIFFERENCE_NORMALIZED,
    INVERSE_VARIANCE,
    MAXIMUM_PROBABILITY,
    SUM_AVERAGE,
    SUM_ENTROPY,
    SUM_OF_SQUARES,
}

impl GLCMFeature {
    pub fn index(&self) -> usize {
        *self as usize
    }
}

/// core mapping routing
pub fn map_glcm(to_calculate:&HashSet<GLCMFeature>, dims:&[usize], image:&[u16], features:&mut [f64], angles:&[[i32;3]], n_bins:usize, kernel_radius:usize, restricted_coords:&[[i32;3]], max_threads:Option<usize>, progress:Arc<AtomicUsize>) {

    println!("dims = {:?}",dims);
    println!("image = {}",image.len());

    let n_features = 24;

    // number of angles to calculate a glcm. One glcm is calculated per angle
    let n_angles = angles.len();

    // size and shape of the input image
    let x_dims = ArrayDim::from_shape(&dims);

    // size and shape of the glcm buffer over all angles
    let glcm_dims = ArrayDim::from_shape(&[n_bins,n_bins,n_angles]);

    // constant values used for feature calculations
    let kvals:Vec<_> = (2..=2*n_bins).map(|x| x as f64).collect();
    let kvald:Vec<_> = (0..n_bins).map(|x| x as f64).collect();

    // small positive number to avoid divisions by 0
    let eps = f64::EPSILON;

    let mut builder = rayon::ThreadPoolBuilder::new();
    let thread_pool = if let Some(max_threads) = max_threads {
        builder.num_threads(max_threads).build().unwrap()
    }else {
        builder.build().unwrap()
    };


    thread_pool.install(|| {



    // main par-for loop for feature calculations. Each thread works on a single output feature vector (feature_set)
    features.par_chunks_exact_mut(n_features).enumerate().for_each(|(g_idx,feature_set)|{

        // if this voxel is 0, then skip all the hard work. We assume voxels that are 0 are masked out
        if image[g_idx] == 0 {
            feature_set.fill(0.);
            progress.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // calculate the i,j,k coordinate of the input image voxel
        let [i, j, k,..] = x_dims.calc_idx(g_idx);
        let i = i as i32;
        let j = j as i32;
        let k = k as i32;

        // option to restrict map generation to only a set of coordinates

        let in_restricted = restricted_coords
            .iter()
            .any(|&coord| coord[0] == i && coord[1] == j && coord[2] == k);

        // return is we are restricting the coordinates for debugging and this coordinate isn't listed
        if !in_restricted && !restricted_coords.is_empty() {
            feature_set.fill(0.);
            progress.fetch_add(1, Ordering::Relaxed);
            return;
        }

        /* ------------------------ */
        /*  NEIGHBOR CALCULATIONS   */
        /* ------------------------ */

        // allocate a buffer for storing voxel neighbor addresses over all angles
        let mut angle_address_pairs:Vec<Vec<[usize;2]>> = vec![vec![]; n_angles];

        // main loop for generating address pairs over all angles
        for (angle_idx,r) in angles.iter().enumerate() {

            // grab the buffer of addresses that we want to add to for this angle
            let address_pairs = angle_address_pairs.get_mut(angle_idx).unwrap();

            // get the range for of the patch (neighborhood) window in native image space
            // let x_range = calc_range_inclusive(patch_radius, r[0]);
            // let y_range = calc_range_inclusive(patch_radius, r[1]);
            // let z_range = calc_range_inclusive(patch_radius, r[2]);

            //println!("x: {:?}",x_range);

            // nested loop over image patch
            // for ii in x_range[0]..=x_range[1] {
            //     for jj in y_range[0]..=y_range[1] {
            //         for kk in z_range[0]..=z_range[1] {
            let pr = kernel_radius as i32;

            for ii in -pr..=pr {
                for jj in -pr..=pr {
                    for kk in -pr..=pr {

                        // virtual root index (can be negative). This is the first neighbor

                        // the neighboring voxel must be in the window
                        if !in_box(kernel_radius, &[ii+r[0],jj+r[1],kk+r[2]]) {
                            continue;
                        }

                        let root_virtual_idx = [i + ii, j + jj, k + kk];

                        // check if the first neighbor is valid. If not, then skip
                        let root_idx = match in_volume(x_dims.shape(), &root_virtual_idx) {
                            Some(idx) => idx,
                            None => continue
                        };

                        // virtual neighbor index. This is the second neighbor
                        let neighbor_virtual_idx = [root_idx[0] as i32 + r[0], root_idx[1] as i32 + r[1], root_idx[2] as i32 + r[2]];

                        // check if the second neighbor is valid. If not, then skip
                        let nei_idx = match in_volume(x_dims.shape(), &neighbor_virtual_idx) {
                            Some(idx) => idx,
                            None => continue
                        };

                        // if both neighbors are valid, then calculate the physical addresses and store it for later
                        address_pairs.push(
                            [
                                x_dims.calc_addr(&root_idx),
                                x_dims.calc_addr(&nei_idx),
                            ]
                        );

                    }
                }
            }

        }

        /* ------------------------ */
        /*    GLCM CALCULATIONS     */
        /* ------------------------ */

        // thread-local GLCM matrix. Every execution thread has its own copy, so no need for locking
        THREAD_GLCM.with(|cell|{

            // borrow the thread-local buffer as the glcm matrix over all angles (3-D array)
            let mut glcm = cell.borrow_mut();

            // ensures that the glcm is the correct size dynamically. For the next iteration in this
            // thread, this should be a no-op
            glcm.resize(glcm_dims.numel(),0.);

            // ensure the glcm is initialized to 0
            glcm.fill(0f64);

            for (angle_idx, addr_pairs) in angle_address_pairs.iter().enumerate() {
                for addr_pair in addr_pairs {


                    // get the neighbor voxel values
                    let n1 = image[addr_pair[0]] as usize;
                    let n2 = image[addr_pair[1]] as usize;

                    // if we encounter a masked-out voxel, then we skip it. Nothing is added to glcm
                    if n1 == 0 || n2 == 0 {
                        continue
                    }

                    // the -1 is because we are assuming 0 means masked-out, so the first valid n value is 1. (1...=n_bins)
                    let glcm_addr = glcm_dims.calc_addr(&[n1 - 1,n2 - 1,angle_idx]);
                    glcm[glcm_addr] += 1.;
                }
            }



            // make the glcm symmetrical (G <- G^T + G). This is because we are only calculating
            // half of the angles

            // chunk the data by angle, then symmetrize the matrix
            glcm.chunks_exact_mut(n_bins*n_bins).for_each(|glcm|{
                symmetrize_in_place_f64(glcm,n_bins)
            });

            // normalize each glcm
            glcm.chunks_exact_mut(n_bins*n_bins).for_each(|glcm|{
                let s:f64 = glcm.iter().sum();
                if s != 0. {
                    glcm.iter_mut().for_each(|x| *x /= s);
                }
            });

            /* ------------------------ */
            /* COEFFICIENT CALCULATIONS */
            /* ------------------------ */

            // calculate marginal row and col probabilities
            let mut px = vec![0f64;n_bins*n_angles];
            let mut py = vec![0f64;n_bins*n_angles];
            px.chunks_exact_mut(n_bins).zip(glcm.chunks_exact(n_bins*n_bins)).for_each(|(px,glcm)|{
                calc_marginal_row_prob(glcm,n_bins,px);
            });
            py.chunks_exact_mut(n_bins).zip(glcm.chunks_exact(n_bins*n_bins)).for_each(|(py,glcm)|{
                calc_marginal_col_prob(glcm,n_bins,py);
            });

            // calculate sigma for px and py
            let sig_px:Vec<f64> = px.chunks_exact(n_bins).map(|px| std_dev(&px)).collect();
            let sig_py:Vec<f64> = py.chunks_exact(n_bins).map(|py| std_dev(&py)).collect();

            // calculate mean values
            let mut ux = vec![0f64;n_angles];
            let mut uy = vec![0f64;n_angles];
            calc_mean_gray_intensity(&px,n_bins,&mut ux);
            calc_mean_gray_intensity(&py,n_bins,&mut uy);

            // calculate p_x+y and p_x-y
            let n_pxpy = 2 * n_bins - 1;
            let mut pxpy = vec![0f64;n_pxpy*n_angles];
            let mut pxmy = vec![0f64;n_bins*n_angles];
            glcm.chunks_exact(n_bins*n_bins)
                .zip(pxpy.chunks_exact_mut(n_pxpy))
                .zip(pxmy.chunks_exact_mut(n_bins))
                .for_each(|((glcm,pxpy),pxmy)|{
                    calc_marginals_inplace(glcm,n_bins,pxpy,pxmy)
                });

            // calculate joint entropy
            let mut hxy = vec![0f64;n_angles];
            glcm.chunks_exact(n_bins*n_bins).zip(hxy.iter_mut()).for_each(|(glcm,hxy)|{
                *hxy = calc_joint_entropy(glcm,eps);
            });

            /* ------------------------ */
            /* FEATURE CALCULATIONS */
            /* ------------------------ */

            use GLCMFeature::*;

            // 1: AUTOCORRELATION
            if to_calculate.contains(&AUTO_CORRELATION) {
                feature_set[AUTO_CORRELATION as usize] = glcm.chunks_exact(n_bins*n_bins).zip(ux.iter().zip(&uy)).zip(sig_px.iter().zip(&sig_py)).map(|((glcm,(&ux,&uy)),(&s_px,&s_py))|{
                    calc_auto_correlation(glcm, n_bins)
                }).sum::<f64>() / n_angles as f64;
            }

            // 2: JOINT AVERAGE
            if to_calculate.contains(&JOINT_AVERAGE) {
                feature_set[JOINT_AVERAGE as usize] = ux.iter().sum::<f64>() / n_angles as f64;
            }

            if to_calculate.contains(&CLUSTER_PROMINENCE) || to_calculate.contains(&CLUSTER_SHADE) ||
                to_calculate.contains(&CLUSTER_TENDENCY) || to_calculate.contains(&CLUSTER_PROMINENCE) {
                // features 3 - 6
                let mut prom = 0.;
                let mut shade = 0.;
                let mut tend = 0.;
                let mut cont = 0.;
                glcm.chunks_exact(n_bins*n_bins).zip(ux.iter().zip(&uy)).for_each(|(glcm,(&ux,&uy))|{
                    let [prom_,shade_,tend_,cont_] = calc_cluster(glcm, n_bins, ux, uy);
                    prom += prom_;
                    shade += shade_;
                    tend += tend_;
                    cont += cont_;
                });
                // mean over all angles
                prom /= n_angles as f64;
                shade /= n_angles as f64;
                tend /= n_angles as f64;
                cont /= n_angles as f64;

                // 3: CLUSTER PROMINENCE
                feature_set[CLUSTER_PROMINENCE as usize] = prom;

                // 4: CLUSTER SHADE
                feature_set[CLUSTER_SHADE as usize] = shade;

                // 5: CLUSTER TENDENCY
                feature_set[CLUSTER_TENDENCY as usize] = tend;

                // 6: CONTRAST
                feature_set[CONTRAST as usize] = cont;
            }

            // 7: CORRELATION
            if to_calculate.contains(&CORRELATION) {
                feature_set[CORRELATION as usize] = glcm.chunks_exact(n_bins*n_bins).zip(ux.iter().zip(&uy)).zip(sig_px.iter().zip(&sig_py)).map(|((glcm,(&ux,&uy)),(&s_px,&s_py))|{
                    calc_correlation(glcm, n_bins, ux, uy, eps)
                }).sum::<f64>() / n_angles as f64;
            }

            // 8: DIFFERENCE AVERAGE
            if to_calculate.contains(&DIFFERENCE_AVERAGE) {
                feature_set[DIFFERENCE_AVERAGE as usize] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                    calc_difference_average(pxmy, &kvald)
                }).sum::<f64>() / n_angles as f64;
            }

            // 9: DIFFERENCE ENTROPY
            if to_calculate.contains(&DIFFERENCE_ENTROPY) {
                feature_set[DIFFERENCE_ENTROPY as usize] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                    calc_difference_entropy(pxmy, eps)
                }).sum::<f64>() / n_angles as f64;
            }

            // 10: DIFFERENCE VARIANCE
            if to_calculate.contains(&DIFFERENCE_VARIANCE) {
                feature_set[DIFFERENCE_VARIANCE as usize] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                    calc_difference_variance(pxmy, &kvald)
                }).sum::<f64>() / n_angles as f64;
            }

            // 11: JOINT ENERGY
            if to_calculate.contains(&JOINT_ENERGY) {
                feature_set[JOINT_ENERGY as usize] = glcm.chunks_exact(n_bins*n_bins).map(|glcm|{
                    calc_joint_energy(glcm)
                }).sum::<f64>() / n_angles as f64;
            }

            // 12: JOINT ENTROPY
            if to_calculate.contains(&JOINT_ENTROPY) {
                feature_set[JOINT_ENTROPY as usize] = glcm.chunks_exact(n_bins*n_bins).map(|glcm|{
                    calc_joint_entropy(glcm, eps)
                }).sum::<f64>() / n_angles as f64;
            }

            // 13: IMC1
            if to_calculate.contains(&IMC1) {
                feature_set[IMC1 as usize] = glcm.chunks_exact(n_bins*n_bins).zip(&hxy)
                    .zip(px.chunks_exact(n_bins).zip(py.chunks_exact(n_bins)))
                    .map(|((glcm,&hxy),(px,py))|{
                        calc_imc1(glcm,n_bins,px,py,hxy,eps)
                    }).sum::<f64>() / n_angles as f64;
            }

            // 14: IMC2
            if to_calculate.contains(&IMC2) {
                feature_set[IMC2 as usize] = hxy.iter()
                    .zip(px.chunks_exact(n_bins).zip(py.chunks_exact(n_bins)))
                    .map(|(&hxy,(px,py))|{
                        calc_imc2(px,py,hxy,eps)
                    }).sum::<f64>() / n_angles as f64;
            }

            // 15: INVERSE DIFFERENCE MOMENT
            if to_calculate.contains(&INVERSE_DIFFERENCE_MOMENT) {
                feature_set[INVERSE_DIFFERENCE_MOMENT as usize] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                    calc_idm(pxmy,&kvald)
                }).sum::<f64>() / n_angles as f64;
            }

            // 16: MAXIMUM CORRELATION COEFFICIENT (the expensive one)
            if to_calculate.contains(&MAXIMUM_CORRELATION_COEFFICIENT) {
                let mut scratch_mat = vec![0.;n_bins*n_bins];
                feature_set[MAXIMUM_CORRELATION_COEFFICIENT as usize] = glcm.chunks_exact(n_bins*n_bins)
                    .zip(px.chunks_exact(n_bins).zip(py.chunks_exact(n_bins)))
                    .map(|(glcm,(px,py))|{
                        calc_mcc(glcm,&mut scratch_mat,n_bins,px,py,eps)
                        //0.
                    }).sum::<f64>() / n_angles as f64;
            }

            // 17: INVERSE DIFFERENCE MOMENT NORMALIZED
            if to_calculate.contains(&INVERSE_DIFFERENCE_MOMENT_NORMALIZED) {
                feature_set[INVERSE_DIFFERENCE_MOMENT_NORMALIZED as usize] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                    calc_idmn(pxmy,&kvald,n_bins)
                }).sum::<f64>() / n_angles as f64;
            }

            // 18: INVERSE DIFFERENCE
            if to_calculate.contains(&INVERSE_DIFFERENCE) {
                feature_set[INVERSE_DIFFERENCE as usize] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                    calc_id(pxmy,&kvald)
                }).sum::<f64>() / n_angles as f64;
            }

            // 19: INVERSE DIFFERENCE NORMALIZED
            if to_calculate.contains(&INVERSE_DIFFERENCE_NORMALIZED) {
                feature_set[INVERSE_DIFFERENCE_NORMALIZED as usize] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                    calc_idn(pxmy,&kvald,n_bins)
                }).sum::<f64>() / n_angles as f64;
            }

            // 20: INVERSE VARIANCE
            if to_calculate.contains(&INVERSE_VARIANCE) {
                feature_set[INVERSE_VARIANCE as usize] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                    calc_inverse_var(pxmy,&kvald)
                }).sum::<f64>() / n_angles as f64;
            }

            // 21: MAXIMUM PROBABILITY
            if to_calculate.contains(&MAXIMUM_PROBABILITY) {
                feature_set[MAXIMUM_PROBABILITY as usize] = glcm.chunks_exact(n_bins*n_bins).map(|glcm|{
                    calc_max_prob(glcm)
                }).sum::<f64>() / n_angles as f64;
            }

            // 22: SUM AVERAGE
            if to_calculate.contains(&SUM_AVERAGE) {
                feature_set[SUM_AVERAGE as usize] = pxpy.chunks_exact(n_pxpy).map(|pxpy|{
                    calc_sum_average(pxpy,&kvals)
                }).sum::<f64>() / n_angles as f64;
            }

            // 23: SUM ENTROPY
            if to_calculate.contains(&SUM_ENTROPY) {
                feature_set[SUM_ENTROPY as usize] = pxpy.chunks_exact(n_pxpy).map(|pxpy|{
                    calc_sum_entropy(pxpy,eps)
                }).sum::<f64>() / n_angles as f64;
            }

            // 24: SUM OF SQUARES
            if to_calculate.contains(&SUM_OF_SQUARES) {
                feature_set[SUM_OF_SQUARES as usize] = glcm.chunks_exact(n_bins*n_bins).zip(&ux).map(|(glcm,&ux)|{
                    calc_sum_of_squares(glcm,n_bins,ux)
                }).sum::<f64>() / n_angles as f64;
            }

            // increment the progress counter
            progress.fetch_add(1,Ordering::Relaxed);

        });


    });

    });

}
