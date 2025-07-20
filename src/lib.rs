use std::cell::RefCell;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use array_lib::ArrayDim;
use crate::subr::{auto_correlation, calc_idm, calc_imc, calc_joint_entropy, calc_marginal_col_prob, calc_marginal_row_prob, calc_marginals_inplace, calc_mcc, calc_mean_gray_intensity, cluster, correlation, difference_average, difference_entropy, difference_variance, get_range_inclusive, in_bounds, joint_energy, joint_entropy, std_dev, symmetrize_in_place_f64};
use rayon::prelude::*;

mod core;
mod subr;

// this allocates a dynamically sized array to store GLCM data (1 per thread)
// once a thread wants to write data to it, it will resize it appropriately
thread_local! {
    static THREAD_GLCM: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}


#[cfg(test)]
mod tests {
    use std::time::Instant;
    use array_lib::ArrayDim;
    use array_lib::io_nifti::write_nifti;
    use rand;
    use rand::Rng;
    use rayon::prelude::*;
    use crate::subr::{auto_correlation, calc_joint_entropy, calc_marginal_col_prob, calc_marginal_row_prob, calc_marginals_inplace, calc_mean_gray_intensity, cluster, correlation, difference_average, difference_entropy, difference_variance, get_range_inclusive, in_bounds, joint_energy, joint_entropy, std_dev, symmetrize_in_place_f64};
    use crate::{change_dims, map_glcm, THREAD_GLCM};

    #[test]
    fn test_map_speed() {

        let n = 100;
        let n_features = 22;
        let dims = [n,n,n];
        let o_dims = [n_features,n,n,n];
        let patch_radius = 2;
        let n_bins = 64;

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
        map_glcm(&dims, &x, 22, &mut features, &angles, n_bins, patch_radius);
        let dur = now.elapsed();

        let o_dim = ArrayDim::from_shape(&[n,n,n,n_features]);
        let mut out = o_dim.alloc(0f64);

        println!("swapping dims ...");
        change_dims(&dims,n_features,&features,&mut out);
        println!("writing out ...");
        write_nifti("test",&out,o_dim);

        println!("Elapsed: {:.2?} sec", dur.as_secs_f64());
    }


    #[test]
    fn test_img() {

        // about 10us per voxel to calculate glcm with patch radius of 2 for one direction

        let dims = [3,3,3];
        let patch_radius = 2;
        let patch_size = [2*patch_radius + 1;3];
        let n_bins:usize = 5;
        let n_features = 4;
        let x_dims = ArrayDim::from_shape(&dims);
        let o_dims = ArrayDim::from_shape(&[n_features, dims[0],dims[1],dims[2]]);
        let p_dims = ArrayDim::from_shape(&patch_size);
        let mut x = x_dims.alloc(0u16);
        let mut out = o_dims.alloc(0f64);

        let eps = f64::EPSILON;

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

        //let mut x = ArrayDim::from_shape(&dims).alloc(0u16);
        // x.copy_from_slice(
        //     &[1,1,2,2,4,5,3,4,2,3,2,1,4,2,5,3,1,1,5,3,2,1,1,5,4,4,1]
        // );
        x.copy_from_slice(
            &[0,1,2,0,4,5,3,4,2,3,2,1,4,2,5,3,1,1,5,3,2,1,1,5,4,4,0]
        );

        // let mut rng = rand::rng();
        // x.iter_mut().for_each(|x| *x = rng.random_range(1..=n_bins as u16));
        let x = x;

        println!("x = {:?}",x);

        // maybe use some memory pool that serves glcm matrices to use for calculations
        let glcm_dims = ArrayDim::from_shape(&[n_bins,n_bins,n_angles]);
        //let glcm_scratch = Arc::new(Mutex::new(vec![0usize;glcm_dims.numel()]));

        let now_outer = Instant::now();

        (&mut out).par_chunks_exact_mut(n_features).enumerate().for_each(|(g_idx,feature_set)|{

            // if this voxel is 0, then skip all of this hard work
            if x[g_idx] == 0 {
                feature_set.fill(f64::NAN);
                return;
            }

            let now = Instant::now();

            // this is the i,j,k coordinate of this image voxel
            let [i,j,k,..] = x_dims.calc_idx(g_idx);
            let i = i as i32;
            let j = j as i32;
            let k = k as i32;

            // just for testing
            if i != 1 || j != 1 || k != 1 {
                return
            }

            // this is where we store the neighbor addresses to calculate glcm over all angles
            let mut angle_pairs:Vec<Vec<[usize;2]>> = vec![vec![];n_angles];

            // iterate over all angles
            for (angle_idx,r) in angles.iter().enumerate() {

                // grab the vec of addresses that we want to add to for this angle
                let these_angle_pairs = angle_pairs.get_mut(angle_idx).unwrap();

                // get the range for of the neighbor window. This is the image patch that makes sense
                // to sample
                let x_range = get_range_inclusive(patch_size[0],r[0]);
                let y_range = get_range_inclusive(patch_size[1],r[1]);
                let z_range = get_range_inclusive(patch_size[2],r[2]);

                // simple loop over the valid image patch
                for ii in x_range[0]..=x_range[1] {
                    for jj in y_range[0]..=y_range[1] {
                        for kk in z_range[0]..=z_range[1] {

                            // virtual root index (can be negative). This is the first neighbor
                            let root_vidx = [i + ii, j + jj, k + kk];

                            // check if the first neighbor is valid
                            let root_idx = match in_bounds(x_dims.shape(), &root_vidx) {
                                Some(idx) => idx,
                                None => continue
                            };

                            // virtual neighbor index. This is the second neighbor
                            let nei_vidx = [root_idx[0] as i32 + r[0], root_idx[1] as i32 + r[1], root_idx[2] as i32 + r[2]];

                            // check if the second neighbor is valid
                            let nei_idx = match in_bounds(x_dims.shape(), &nei_vidx) {
                                Some(idx) => idx,
                                None => continue
                            };

                            these_angle_pairs.push(
                                [
                                    x_dims.calc_addr(&root_idx),
                                    x_dims.calc_addr(&nei_idx),
                                ]
                            );

                        }
                    }
                }
            }

            //panic!("angle pairs {:?}",angle_pairs.len());
            // thread-local GLCM matrix
            THREAD_GLCM.with(|cell|{

                // borrow the thread-local buffer as the glcm matrix over all angles (3-D array)
                let mut glcm = cell.borrow_mut();

                // ensures that glcm is the correct size dynamically. For the next iteration in this
                // thread, this should be a no-op
                glcm.resize(glcm_dims.numel(),0.);

                // make sure the glcm is initialized to 0
                glcm.fill(0f64);

                for (angle_idx, addr_pairs) in angle_pairs.iter().enumerate() {
                    for addr_pair in addr_pairs {
                        // the -1 is because we are assuming 0 means masked-out, so the first valid value is 1..=n_bins

                        let n1 = x[addr_pair[0]] as usize;
                        let n2 = x[addr_pair[1]] as usize;
                        if n1 == 0 || n2 == 0 {
                            continue
                        }

                        let glcm_addr = glcm_dims.calc_addr(&[n1 - 1,n2 - 1,angle_idx]);
                        glcm[glcm_addr] += 1.;
                    }
                }

                // make the glcm symmetrical (G <- G^T + G)
                // chunk the data by angle, then symmetrize the matrix
                glcm.chunks_exact_mut(n_bins*n_bins).for_each(|glcm|{
                    symmetrize_in_place_f64(glcm,n_bins)
                });

                // normalize the matrix
                glcm.chunks_exact_mut(n_bins*n_bins).for_each(|glcm|{
                    let s:f64 = glcm.iter().sum();
                    glcm.iter_mut().for_each(|x| *x /= s);
                });

                // calculate coefficients

                // calculate marginal row probabilities
                // calculate marginal col probabilities
                let mut px = vec![0f64;n_bins*n_angles];
                let mut py = vec![0f64;n_bins*n_angles];

                px.chunks_exact_mut(n_bins).zip(glcm.chunks_exact(n_bins*n_bins)).for_each(|(px,glcm)|{
                    calc_marginal_row_prob(glcm,n_bins,px);
                });
                py.chunks_exact_mut(n_bins).zip(glcm.chunks_exact(n_bins*n_bins)).for_each(|(py,glcm)|{
                    calc_marginal_col_prob(glcm,n_bins,py);
                });

                let sig_px:Vec<f64> = px.chunks_exact(n_bins).map(|px| std_dev(&px)).collect();
                let sig_py:Vec<f64> = py.chunks_exact(n_bins).map(|py| std_dev(&py)).collect();

                let mut ux = vec![0f64;n_angles];
                let mut uy = vec![0f64;n_angles];

                calc_mean_gray_intensity(&px,n_bins,&mut ux);
                calc_mean_gray_intensity(&py,n_bins,&mut uy);

                //println!("ux = {:?}",&ux);
                //println!("uy = {:?}",&uy);

                let n_pxpy = 2 * n_bins - 1;
                let mut pxpy = vec![0f64;n_pxpy*n_angles];
                let mut pxmy = vec![0f64;n_bins*n_angles];

                glcm.chunks_exact(n_bins*n_bins)
                    .zip(pxpy.chunks_exact_mut(n_pxpy))
                    .zip(pxmy.chunks_exact_mut(n_bins))
                    .for_each(|((glcm,pxpy),pxmy)|{
                        calc_marginals_inplace(glcm,n_bins,pxpy,pxmy)
                    });

                //println!("pxpy = {:?}",&pxpy[0..n_pxpy]);
                //println!("pxmy = {:?}",&pxmy[0..n_bins]);

                let mut hxy = vec![0f64;n_angles];
                glcm.chunks_exact(n_bins*n_bins).zip(hxy.iter_mut()).for_each(|(glcm,hxy)|{
                    *hxy = calc_joint_entropy(glcm,eps);
                });

                //println!("hxy = {:?}",&hxy);

                let kvals:Vec<_> = (2..=2*n_bins).map(|x| x as f64).collect();
                let kvald:Vec<_> = (0..n_bins).map(|x| x as f64).collect();

                //println!("kvals = {:?}",&kvals);
                //println!("kvald = {:?}",&kvald);


                // calculate features from glcm

                // feature 1 AUTOCORRELATION

                let mut auto_corr = 0.;
                let mut corr = 0.;
                glcm.chunks_exact(n_bins*n_bins).zip(ux.iter().zip(&uy)).zip(sig_px.iter().zip(&sig_py)).for_each(|((glcm,(&ux,&uy)),(&s_px,&s_py))|{
                    auto_corr += auto_correlation(glcm, n_bins);
                    corr += correlation(auto_corr,ux,uy,s_px,s_py);
                });
                // mean over all angles
                auto_corr /= n_angles as f64;
                corr /= n_angles as f64;
                feature_set[0] = auto_corr;
                feature_set[6] = corr;
                // feature 2 JOINT AVERAGE
                feature_set[1] = ux.iter().sum::<f64>() / n_angles as f64;

                // features 3, 4, 5, 6
                // CLUSTER PROMINENCE, CLUSTER SHADE, CLUSTER TENDENCY, CONTRAST

                let mut prom = 0.;
                let mut shade = 0.;
                let mut tend = 0.;
                let mut cont = 0.;
                glcm.chunks_exact(n_bins*n_bins).zip(ux.iter().zip(&uy)).for_each(|(glcm,(&ux,&uy))|{
                    let [prom_,shade_,tend_,cont_] = cluster(glcm,n_bins,ux,uy);
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

                feature_set[2] = prom;
                feature_set[3] = shade;
                feature_set[4] = tend;
                feature_set[5] = cont;

                // difference average
                feature_set[7] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                    difference_average(pxmy,&kvald)
                }).sum::<f64>() / n_angles as f64;

                // difference entropy
                feature_set[8] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                    difference_entropy(pxmy,eps)
                }).sum::<f64>() / n_angles as f64;

                // difference variance
                feature_set[9] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                    difference_variance(pxmy,&kvald)
                }).sum::<f64>() / n_angles as f64;

                // joint energy
                feature_set[10] = glcm.chunks_exact(n_bins*n_bins).map(|glcm|{
                    joint_energy(glcm)
                }).sum::<f64>() / n_angles as f64;

                // joint entropy
                feature_set[11] = glcm.chunks_exact(n_bins*n_bins).map(|glcm|{
                    joint_entropy(glcm,eps)
                }).sum::<f64>() / n_angles as f64;






            });


            let dur = now.elapsed();

            // calculate stuff with glcm
            //println!("glcm took: {} us",dur.as_micros());

        });

        let dur = now_outer.elapsed();
        println!("took {} sec",dur.as_secs_f64());


    }
}

pub fn map_glcm(dims:&[usize],image:&[u16],n_features:usize,features:&mut [f64],angles:&[[i32;3]],n_bins:usize,patch_radius:usize) {

    let n_angles = angles.len();
    let patch_size = [2*patch_radius + 1;3];
    let x_dims = ArrayDim::from_shape(&dims);
    let glcm_dims = ArrayDim::from_shape(&[n_bins,n_bins,n_angles]);

    let eps = f64::EPSILON;
    let mut count = Arc::new(Mutex::new(0));

    features.par_chunks_exact_mut(n_features).enumerate().for_each(|(g_idx,feature_set)|{

        // if this voxel is 0, then skip all of this hard work
        if image[g_idx] == 0 {
            feature_set.fill(f64::NAN);
            return;
        }

        // this is the i,j,k coordinate of this image voxel
        let [i,j,k,..] = x_dims.calc_idx(g_idx);
        let i = i as i32;
        let j = j as i32;
        let k = k as i32;

        // this is where we store the neighbor addresses to calculate glcm over all angles
        let mut angle_pairs:Vec<Vec<[usize;2]>> = vec![vec![];n_angles];

        // iterate over all angles
        for (angle_idx,r) in angles.iter().enumerate() {

            // grab the vec of addresses that we want to add to for this angle
            let these_angle_pairs = angle_pairs.get_mut(angle_idx).unwrap();

            // get the range for of the neighbor window. This is the image patch that makes sense
            // to sample
            let x_range = get_range_inclusive(patch_size[0],r[0]);
            let y_range = get_range_inclusive(patch_size[1],r[1]);
            let z_range = get_range_inclusive(patch_size[2],r[2]);

            // simple loop over the valid image patch
            for ii in x_range[0]..=x_range[1] {
                for jj in y_range[0]..=y_range[1] {
                    for kk in z_range[0]..=z_range[1] {

                        // virtual root index (can be negative). This is the first neighbor
                        let root_vidx = [i + ii, j + jj, k + kk];

                        // check if the first neighbor is valid
                        let root_idx = match in_bounds(x_dims.shape(), &root_vidx) {
                            Some(idx) => idx,
                            None => continue
                        };

                        // virtual neighbor index. This is the second neighbor
                        let nei_vidx = [root_idx[0] as i32 + r[0], root_idx[1] as i32 + r[1], root_idx[2] as i32 + r[2]];

                        // check if the second neighbor is valid
                        let nei_idx = match in_bounds(x_dims.shape(), &nei_vidx) {
                            Some(idx) => idx,
                            None => continue
                        };

                        these_angle_pairs.push(
                            [
                                x_dims.calc_addr(&root_idx),
                                x_dims.calc_addr(&nei_idx),
                            ]
                        );

                    }
                }
            }
        }

        // thread-local GLCM matrix
        THREAD_GLCM.with(|cell|{

            // borrow the thread-local buffer as the glcm matrix over all angles (3-D array)
            let mut glcm = cell.borrow_mut();

            // ensures that glcm is the correct size dynamically. For the next iteration in this
            // thread, this should be a no-op
            glcm.resize(glcm_dims.numel(),0.);

            // make sure the glcm is initialized to 0
            glcm.fill(0f64);

            for (angle_idx, addr_pairs) in angle_pairs.iter().enumerate() {
                for addr_pair in addr_pairs {
                    // the -1 is because we are assuming 0 means masked-out, so the first valid value is 1..=n_bins

                    let n1 = image[addr_pair[0]] as usize;
                    let n2 = image[addr_pair[1]] as usize;
                    if n1 == 0 || n2 == 0 {
                        continue
                    }

                    let glcm_addr = glcm_dims.calc_addr(&[n1 - 1,n2 - 1,angle_idx]);
                    glcm[glcm_addr] += 1.;
                }
            }

            // make the glcm symmetrical (G <- G^T + G)
            // chunk the data by angle, then symmetrize the matrix
            glcm.chunks_exact_mut(n_bins*n_bins).for_each(|glcm|{
                symmetrize_in_place_f64(glcm,n_bins)
            });

            // normalize the matrix
            glcm.chunks_exact_mut(n_bins*n_bins).for_each(|glcm|{
                let s:f64 = glcm.iter().sum();
                glcm.iter_mut().for_each(|x| *x /= s);
            });


            // calculate coefficients

            // calculate marginal row and col probabilities
            let mut px = vec![0f64;n_bins*n_angles];
            let mut py = vec![0f64;n_bins*n_angles];

            px.chunks_exact_mut(n_bins).zip(glcm.chunks_exact(n_bins*n_bins)).for_each(|(px,glcm)|{
                calc_marginal_row_prob(glcm,n_bins,px);
            });
            py.chunks_exact_mut(n_bins).zip(glcm.chunks_exact(n_bins*n_bins)).for_each(|(py,glcm)|{
                calc_marginal_col_prob(glcm,n_bins,py);
            });

            let sig_px:Vec<f64> = px.chunks_exact(n_bins).map(|px| std_dev(&px)).collect();
            let sig_py:Vec<f64> = py.chunks_exact(n_bins).map(|py| std_dev(&py)).collect();

            let hx:Vec<f64> = px.chunks_exact(n_bins).map(|px| joint_entropy(px,eps)).collect();
            let hy:Vec<f64> = py.chunks_exact(n_bins).map(|py| joint_entropy(py,eps)).collect();

            let mut ux = vec![0f64;n_angles];
            let mut uy = vec![0f64;n_angles];

            calc_mean_gray_intensity(&px,n_bins,&mut ux);
            calc_mean_gray_intensity(&py,n_bins,&mut uy);

            let n_pxpy = 2 * n_bins - 1;
            let mut pxpy = vec![0f64;n_pxpy*n_angles];
            let mut pxmy = vec![0f64;n_bins*n_angles];

            glcm.chunks_exact(n_bins*n_bins)
                .zip(pxpy.chunks_exact_mut(n_pxpy))
                .zip(pxmy.chunks_exact_mut(n_bins))
                .for_each(|((glcm,pxpy),pxmy)|{
                    calc_marginals_inplace(glcm,n_bins,pxpy,pxmy)
                });

            let mut hxy = vec![0f64;n_angles];
            glcm.chunks_exact(n_bins*n_bins).zip(hxy.iter_mut()).for_each(|(glcm,hxy)|{
                *hxy = calc_joint_entropy(glcm,eps);
            });

            let kvals:Vec<_> = (2..=2*n_bins).map(|x| x as f64).collect();
            let kvald:Vec<_> = (0..n_bins).map(|x| x as f64).collect();

            // calculate features from glcm

            // feature 1 AUTOCORRELATION
            let mut auto_corr = 0.;
            let mut corr = 0.;
            glcm.chunks_exact(n_bins*n_bins).zip(ux.iter().zip(&uy)).zip(sig_px.iter().zip(&sig_py)).for_each(|((glcm,(&ux,&uy)),(&s_px,&s_py))|{
                auto_corr += auto_correlation(glcm, n_bins);
                corr += correlation(auto_corr,ux,uy,s_px,s_py);
            });
            // mean over all angles
            auto_corr /= n_angles as f64;
            corr /= n_angles as f64;
            feature_set[0] = auto_corr;
            feature_set[6] = corr;
            // feature 2 JOINT AVERAGE
            feature_set[1] = ux.iter().sum::<f64>() / n_angles as f64;

            // features 3, 4, 5, 6
            // CLUSTER PROMINENCE, CLUSTER SHADE, CLUSTER TENDENCY, CONTRAST

            let mut prom = 0.;
            let mut shade = 0.;
            let mut tend = 0.;
            let mut cont = 0.;
            glcm.chunks_exact(n_bins*n_bins).zip(ux.iter().zip(&uy)).for_each(|(glcm,(&ux,&uy))|{
                let [prom_,shade_,tend_,cont_] = cluster(glcm,n_bins,ux,uy);
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

            feature_set[2] = prom;
            feature_set[3] = shade;
            feature_set[4] = tend;
            feature_set[5] = cont;

            // difference average
            feature_set[7] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                difference_average(pxmy,&kvald)
            }).sum::<f64>() / n_angles as f64;

            // difference entropy
            feature_set[8] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                difference_entropy(pxmy,eps)
            }).sum::<f64>() / n_angles as f64;

            // difference variance
            feature_set[9] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                difference_variance(pxmy,&kvald)
            }).sum::<f64>() / n_angles as f64;

            // joint energy
            feature_set[10] = glcm.chunks_exact(n_bins*n_bins).map(|glcm|{
                joint_energy(glcm)
            }).sum::<f64>() / n_angles as f64;

            // joint entropy
            feature_set[11] = glcm.chunks_exact(n_bins*n_bins).map(|glcm|{
                joint_entropy(glcm,eps)
            }).sum::<f64>() / n_angles as f64;

            // imc values
            let mut imc1 = 0.;
            let mut imc2 = 0.;
            glcm.chunks_exact(n_bins*n_bins)
            .zip(px.chunks_exact(n_bins).zip(py.chunks_exact(n_bins)))
            .for_each(|(glcm,(px,py))|{
                let [imc1_,imc2_] = calc_imc(glcm,n_bins,px,py,eps);
                imc1 += imc1_;
                imc2 += imc2_;
            });
            feature_set[12] = imc1 / n_angles as f64;
            feature_set[13] = imc2 / n_angles as f64;

            // inverse difference moment
            feature_set[14] = pxmy.chunks_exact(n_bins).map(|pxmy|{
                calc_idm(pxmy,&kvald)
            }).sum::<f64>() / n_angles as f64;

            let mut scratch_mat = vec![0.;n_bins*n_bins];
            feature_set[15] = glcm.chunks_exact(n_bins*n_bins)
            .zip(px.chunks_exact(n_bins).zip(py.chunks_exact(n_bins)))
            .map(|(glcm,(px,py))|{
                calc_mcc(glcm,&mut scratch_mat,n_bins,px,py)
            }).sum::<f64>() / n_angles as f64;

        });

        *count.lock().unwrap() += 1;


    });


    println!("count = {}",count.lock().unwrap());


}


/// swaps the feature dims to the last dimension
fn change_dims(dims:&[usize],n_features:usize,x:&[f64],y:&mut [f64]) {
    let i_dims = ArrayDim::from_shape(&[n_features,dims[0],dims[1],dims[2]]);
    let o_dims = ArrayDim::from_shape(&[dims[0],dims[1],dims[2],n_features]);
    y.par_iter_mut().enumerate().for_each(|(i,y)|{
        let [i,j,k,f,..] = o_dims.calc_idx(i);
        let addr = i_dims.calc_addr(&[f,i,j,k]);
        *y = x[addr];
    });
}