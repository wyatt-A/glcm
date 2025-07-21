use std::cell::RefCell;
use array_lib::ArrayDim;
use crate::subr::{calc_auto_correlation, calc_id, calc_idm, calc_idmn, calc_idn, calc_imc1, calc_imc2, calc_inverse_var, calc_marginal_col_prob, calc_marginal_row_prob, calc_marginals_inplace, calc_max_prob, calc_mcc, calc_mean_gray_intensity, calc_sum_average, calc_sum_entropy, calc_sum_of_squares, calc_cluster, calc_correlation, calc_difference_average, calc_difference_entropy, calc_difference_variance, calc_range_inclusive, in_volume, calc_joint_energy, calc_joint_entropy, std_dev, symmetrize_in_place_f64, in_box};
use rayon::prelude::*;


// this allocates a dynamically sized array to store GLCM data (1 per thread)
// once a thread wants to write data to it, it will resize it appropriately
thread_local! {
    static THREAD_GLCM: RefCell<Vec<f64>> = RefCell::new(vec![]);
}

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use array_lib::ArrayDim;
    use array_lib::io_nifti::write_nifti;
    use rand;
    use rand::Rng;
    use super::map_glcm;
    use crate::subr::change_dims;

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

        let now = Instant::now();
        map_glcm(&dims, &x, n_features, &mut features, &angles, n_bins, patch_radius,&[]);
        let dur = now.elapsed();

        let o_dim = ArrayDim::from_shape(&[n,n,n,n_features]);
        let mut out = o_dim.alloc(0f64);

        println!("swapping dims ...");
        change_dims(&dims,n_features,&features,&mut out);

        println!("Elapsed: {:.2?} sec", dur.as_secs_f64());

        let vox_per_sec = x.len() as f64 / dur.as_secs_f64();
        println!("computed {vox_per_sec} voxels / sec")
    }

    /// computes GLCM values for a small example to compare to pyradiomic's outputs with the same
    /// parameters
    #[test]
    fn test_pyrad_comparison() {

        let dims = [3,3,3];
        let patch_radius = 1;
        let n_bins:usize = 5;
        let n_features = 24;
        let x_dims = ArrayDim::from_shape(&dims);
        let f_dims = ArrayDim::from_shape(&[n_features, dims[0],dims[1],dims[2]]);
        let mut x = x_dims.alloc(0u16);
        let mut features = f_dims.alloc(0f64);

        // vox to compute (center voxel of 3x3x3)
        let vox = [1,1,1];

        // angles must be on the half-shell, meaning that no additive inverses exist
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

        let n_angles = angles.len();
        x.copy_from_slice(
            &[0,1,2,0,4,5,3,4,2,3,2,1,4,2,5,3,1,1,5,3,2,1,1,5,4,4,0]
        );
        let x = x;

        map_glcm(&dims, &x, n_features, &mut features, &angles, n_bins, patch_radius, &[vox]);

        let mut out = vec![];
        for i in 0..n_features {
            let f_addr = f_dims
                .calc_addr(&[i,vox[0] as usize,vox[1] as usize,vox[2] as usize]);
            println!("{}: {:?}",i+1,features[f_addr]);
            out.push(features[f_addr]);
        }

        let pyrad_features:[f64;24] = [
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
        let error:Vec<_> = out.iter().zip(&pyrad_features).map(|(&y,&x)| x - y).collect();
        let norm_err = error.iter().map(|e| e.powi(2)).sum::<f64>().sqrt();
        println!("norm_err: {:?}", norm_err);
        assert!(error.iter().all(|&e| e < 1.0e-6));

    }
}

pub fn map_glcm(dims:&[usize],image:&[u16],n_features:usize,features:&mut [f64],angles:&[[i32;3]],n_bins:usize,patch_radius:usize,restricted_coords:&[[i32;3]]) {

    // number of angles to calculate a glcm. One glcm is calculated per angle
    let n_angles = angles.len();

    // the patch is the local region over which the glcm is calculated. This is distinct from the angles
    let patch_size = [2*patch_radius + 1;3];

    // size and shape of the input image
    let x_dims = ArrayDim::from_shape(&dims);

    // size and shape of the glcm buffer over all angles
    let glcm_dims = ArrayDim::from_shape(&[n_bins,n_bins,n_angles]);

    // constant values used for feature calculations
    let kvals:Vec<_> = (2..=2*n_bins).map(|x| x as f64).collect();
    let kvald:Vec<_> = (0..n_bins).map(|x| x as f64).collect();

    // small positive number to avoid divisions by 0
    let eps = f64::EPSILON;

    // main par-for loop for feature calculations. Each thread works on a single output feature vector (feature_set)
    features.par_chunks_exact_mut(n_features).enumerate().for_each(|(g_idx,feature_set)|{

        // if this voxel is 0, then skip all the hard work. We assume voxels that are 0 are masked out
        if image[g_idx] == 0 {
            feature_set.fill(0.);
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
            feature_set.fill(f64::NAN);
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
            let pr = patch_radius as i32;

            for ii in -pr..=pr {
                for jj in -pr..=pr {
                    for kk in -pr..=pr {

                        // virtual root index (can be negative). This is the first neighbor

                        // the neighboring voxel must be in the window
                        if !in_box(patch_radius,&[ii+r[0],jj+r[1],kk+r[2]]) {
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

            // 1: AUTOCORRELATION
            feature_set[0] = glcm.chunks_exact(n_bins*n_bins).zip(ux.iter().zip(&uy)).zip(sig_px.iter().zip(&sig_py)).map(|((glcm,(&ux,&uy)),(&s_px,&s_py))|{
                calc_auto_correlation(glcm, n_bins)
            }).sum::<f64>() / n_angles as f64;

            // 2: JOINT AVERAGE
            feature_set[1] = ux.iter().sum::<f64>() / n_angles as f64;

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
            feature_set[2] = prom;

            // 4: CLUSTER SHADE
            feature_set[3] = shade;

            // 5: CLUSTER TENDENCY
            feature_set[4] = tend;

            // 6: CONTRAST
            feature_set[5] = cont;

            // 7: CORRELATION
            feature_set[6] = glcm.chunks_exact(n_bins*n_bins).zip(ux.iter().zip(&uy)).zip(sig_px.iter().zip(&sig_py)).map(|((glcm,(&ux,&uy)),(&s_px,&s_py))|{
                calc_correlation(glcm, n_bins, ux, uy, eps)
            }).sum::<f64>() / n_angles as f64;

            // 8: DIFFERENCE AVERAGE
            feature_set[7] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                calc_difference_average(pxmy, &kvald)
            }).sum::<f64>() / n_angles as f64;

            // 9: DIFFERENCE ENTROPY
            feature_set[8] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                calc_difference_entropy(pxmy, eps)
            }).sum::<f64>() / n_angles as f64;

            // 10: DIFFERENCE VARIANCE
            feature_set[9] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                calc_difference_variance(pxmy, &kvald)
            }).sum::<f64>() / n_angles as f64;

            // 11: JOINT ENERGY
            feature_set[10] = glcm.chunks_exact(n_bins*n_bins).map(|glcm|{
                calc_joint_energy(glcm)
            }).sum::<f64>() / n_angles as f64;

            // 12: JOINT ENTROPY
            feature_set[11] = glcm.chunks_exact(n_bins*n_bins).map(|glcm|{
                calc_joint_entropy(glcm, eps)
            }).sum::<f64>() / n_angles as f64;

            // 13: IMC1
            feature_set[12] = glcm.chunks_exact(n_bins*n_bins).zip(&hxy)
                .zip(px.chunks_exact(n_bins).zip(py.chunks_exact(n_bins)))
                .map(|((glcm,&hxy),(px,py))|{
                    calc_imc1(glcm,n_bins,px,py,hxy,eps)
                }).sum::<f64>() / n_angles as f64;

            // 14: IMC2
            feature_set[13] = hxy.iter()
                .zip(px.chunks_exact(n_bins).zip(py.chunks_exact(n_bins)))
                .map(|(&hxy,(px,py))|{
                    calc_imc2(px,py,hxy,eps)
                }).sum::<f64>() / n_angles as f64;

            // 15: INVERSE DIFFERENCE MOMENT
            feature_set[14] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                calc_idm(pxmy,&kvald)
            }).sum::<f64>() / n_angles as f64;

            // 16: MAXIMUM CORRELATION COEFFICIENT (the expensive one)
            // let mut scratch_mat = vec![0.;n_bins*n_bins];
            // feature_set[15] = glcm.chunks_exact(n_bins*n_bins)
            //     .zip(px.chunks_exact(n_bins).zip(py.chunks_exact(n_bins)))
            //     .map(|(glcm,(px,py))|{
            //         calc_mcc(glcm,&mut scratch_mat,n_bins,px,py,eps)
            //         //0.
            //     }).sum::<f64>() / n_angles as f64;

            // 17: INVERSE DIFFERENCE MOMENT NORMALIZED
            feature_set[16] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                calc_idmn(pxmy,&kvald,n_bins)
            }).sum::<f64>() / n_angles as f64;

            // 18: INVERSE DIFFERENCE
            feature_set[17] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                calc_id(pxmy,&kvald)
            }).sum::<f64>() / n_angles as f64;

            // 19: INVERSE DIFFERENCE NORMALIZED
            feature_set[18] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                calc_idn(pxmy,&kvald,n_bins)
            }).sum::<f64>() / n_angles as f64;

            // 20: INVERSE VARIANCE
            feature_set[19] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                calc_inverse_var(pxmy,&kvald)
            }).sum::<f64>() / n_angles as f64;

            // 21: MAXIMUM PROBABILITY
            feature_set[20] = glcm.chunks_exact(n_bins*n_bins).map(|glcm|{
                calc_max_prob(glcm)
            }).sum::<f64>() / n_angles as f64;

            // 22: SUM AVERAGE
            feature_set[21] = pxpy.chunks_exact(n_pxpy).map(|pxpy|{
                calc_sum_average(pxpy,&kvals)
            }).sum::<f64>() / n_angles as f64;

            // 23: SUM ENTROPY
            feature_set[22] = pxpy.chunks_exact(n_pxpy).map(|pxpy|{
                calc_sum_entropy(pxpy,eps)
            }).sum::<f64>() / n_angles as f64;

            // 24: SUM OF SQUARES
            feature_set[23] = glcm.chunks_exact(n_bins*n_bins).zip(&ux).map(|(glcm,&ux)|{
                calc_sum_of_squares(glcm,n_bins,ux)
            }).sum::<f64>() / n_angles as f64;

        });


    });

}
