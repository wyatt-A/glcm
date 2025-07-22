use num_traits::{ToPrimitive, Zero};
use rayon::prelude::*;

mod glcm;
mod core;
mod subr;
mod ui;

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Instant;
    use array_lib::ArrayDim;
    use array_lib::io_nifti::{read_nifti, write_nifti, write_nifti_with_header};
    use rayon::prelude::*;
    use crate::{discretize_bin_count, discretize_bin_width, generate_angles, n_angles};
    use crate::glcm::map_glcm;
    use crate::subr::change_dims;
    use crate::ui::RadMapOpts;

    #[test]
    fn test_map_accuracy() {

        let (mask,mask_dims,mask_h) = read_nifti::<f64>("B:/ProjectSpace/wa41/RadMapTest/24SD022/orthob2000/MASK.nii");

        let mask:Vec<_> = mask.par_iter().map(|&x| if x == 0. {0u8} else {1u8}).collect();
        let n_vox = mask.iter().filter(|m| **m > 0).count();

        let n_bins = 64;
        let n_features = 24;
        let patch_radius = 1;

        let mut angles = vec![[0,0,0];n_angles(patch_radius)];
        generate_angles(&mut angles,patch_radius);

        let (img,img_dims,img_h) = read_nifti::<f64>("B:/ProjectSpace/wa41/RadMapTest/24SD022/orthob2000/Reg_2000_avg-ds.nii");

        let dims = &img_dims.shape()[0..3];

        let feature_dims = ArrayDim::from_shape(&[n_features,dims[0],dims[1],dims[2]]);
        let out_dims = ArrayDim::from_shape(&[dims[0],dims[1],dims[2],n_features]);
        let mut features = feature_dims.alloc(0f64);

        let mut bins = vec![0;img.len()];
        discretize_bin_count(n_bins,&img,&mask,&mut bins);

        let opts = RadMapOpts::default();

        println!("running glcm ...");
        let now = Instant::now();
        map_glcm(&opts,dims,&bins,n_features,&mut features,&angles,n_bins,patch_radius,&[]);
        let dur = now.elapsed();
        println!("done.");

        println!("processed {n_vox} voxels in {} secs.",dur.as_secs_f64());

        let mut out = out_dims.alloc(0f64);
        change_dims(dims,n_features,&features,&mut out);

        let out_dir = "B:/ProjectSpace/wa41/RadMapTest/24SD022/orthob2000";

        let base = "orthob2000";
        let vol_stride:usize = dims.iter().product();
        for (f,alias) in opts.features() {
            let i = f as usize;
            let vol = &out[i*vol_stride..(i+1) * vol_stride];
            let path = Path::new(out_dir).join(format!("{}{}{}",base,opts.separator(),alias));
            write_nifti_with_header(path,vol,img_dims,&img_h);
        }

    }


    #[test]
    fn test_values() {


        let reference_files = [
            "RadiomicMapping_original_glcm_Autocorrelation_2.nii",
            "RadiomicMapping_original_glcm_ClusterProminence_2.nii",
            "RadiomicMapping_original_glcm_ClusterShade_2.nii",
            "RadiomicMapping_original_glcm_ClusterTendency_2.nii",
            "RadiomicMapping_original_glcm_Contrast_2.nii",
            "RadiomicMapping_original_glcm_Correlation_2.nii",
            "RadiomicMapping_original_glcm_DifferenceAverage_2.nii",
            "RadiomicMapping_original_glcm_DifferenceEntropy_2.nii",
            "RadiomicMapping_original_glcm_DifferenceVariance_2.nii",
            "RadiomicMapping_original_glcm_Id_2.nii",
            "RadiomicMapping_original_glcm_Idm_2.nii",
            "RadiomicMapping_original_glcm_Idmn_2.nii",
            "RadiomicMapping_original_glcm_Idn_2.nii",
            "RadiomicMapping_original_glcm_Imc1_2.nii",
            "RadiomicMapping_original_glcm_Imc2_2.nii",
            "RadiomicMapping_original_glcm_InverseVariance_2.nii",
            "RadiomicMapping_original_glcm_JointAverage_2.nii",
            "RadiomicMapping_original_glcm_JointEnergy_2.nii",
            "RadiomicMapping_original_glcm_JointEntropy_2.nii",
            "RadiomicMapping_original_glcm_MaximumProbability_2.nii",
            "RadiomicMapping_original_glcm_SumEntropy_2.nii",
            "RadiomicMapping_original_glcm_SumSquares_2.nii",
        ];

        let test_files = [
            "orthob2000_auto_correlation.nii",
            "orthob2000_cluster_prominence.nii",
            "orthob2000_cluster_shade.nii",
            "orthob2000_cluster_tendency.nii",
            "orthob2000_contrast.nii",
            "orthob2000_correlation.nii",
            "orthob2000_difference_average.nii",
            "orthob2000_difference_entropy.nii",
            "orthob2000_difference_variance.nii",
            "orthob2000_inverse_difference.nii",
            "orthob2000_inverse_difference_moment.nii",
            "orthob2000_inverse_difference_moment_normalized.nii",
            "orthob2000_inverse_difference_normalized.nii",
            "orthob2000_imc1.nii",
            "orthob2000_imc2.nii",
            "orthob2000_inverse_variance.nii",
            "orthob2000_joint_average.nii",
            "orthob2000_joint_energy.nii",
            "orthob2000_joint_entropy.nii",
            "orthob2000_maximum_probability.nii",
            "orthob2000_sum_entropy.nii",
            "orthob2000_sum_of_squares.nii",
        ];

        let base = Path::new("B:\\ProjectSpace\\wa41\\RadMapTest\\24SD022\\original_results_corrected");

        let vox_reference = reference_files.par_iter().map(|f| {
            let p = base.join(f);
            println!("reading {}",p.display());
            let (refer,rdims,..) = read_nifti::<f64>(p);
            let a = rdims.calc_addr(&[95,101,43]);
            refer[a]
        }).collect::<Vec<_>>();

        let base = Path::new("B:\\ProjectSpace\\wa41\\RadMapTest\\24SD022\\orthob2000");
        let vox_test = test_files.par_iter().map(|f| {
            let p = base.join(f);
            println!("reading {}",p.display());
            let (test,tdims,..) = read_nifti::<f64>(p);
            let a = tdims.calc_addr(&[113,125,112]);
            test[a]
        }).collect::<Vec<_>>();

        let norm_res = vox_reference.iter().zip(vox_test.iter()).map(|(&a,&b)| (a - b).powi(2)).sum::<f64>().sqrt();


        let diff:Vec<_> = vox_reference.iter().zip(vox_test.iter()).map(|(&a,&b)| (a - b).abs()).collect();

        println!("diff^2: {:?}", diff);

        println!("norm_res = {:?}",norm_res);

    }

    #[test]
    fn test_discretize() {

        let x = [1.,3.,10.,11.,15.,62.];
        let m = [0,1,1,1,1,1];

        let mut d = vec![0;x.len()];

        discretize_bin_count(3,&x,&m,&mut d);
        assert_eq!(d,[0, 1, 1, 1, 1, 3]);

        discretize_bin_width(2.,&x,&m,&mut d);
        assert_eq!(d,[0, 1, 5, 5, 7, 31]);

    }

    #[test]
    fn test_angles() {
        let r = 1;
        let n = n_angles(r);
        let mut angles = vec![[0,0,0];n];
        generate_angles(&mut angles,r);

        let expected = [
            [ 1 , 1 , 1],[ 1 , 1 , 0],[ 1 , 1 ,-1],[ 1 , 0 , 1],
            [ 1 , 0 , 0],[ 1,  0 ,-1], [ 1, -1 , 1], [ 1, -1 , 0],
            [ 1 ,-1 ,-1],[ 0 , 1 , 1],[ 0 , 1 , 0],[ 0 , 1 ,-1],
            [ 0,  0 , 1],
        ];

        assert_eq!(angles, expected);
    }


}






/// discretize gray level intensities 'x' into 'n_bins' within some roi 'mask'. The result is
/// written to 'd' as unsigned 16-bit values
fn discretize_bin_count<M:Zero + Copy + Send + Sync>(n_bins:usize, x:&[f64], mask:&[M], d:&mut [u16]) {

    assert!(n_bins > 0);
    assert!(n_bins < u16::MAX as usize);
    assert_eq!(mask.len(), x.len());
    assert_eq!(d.len(), x.len());

    // determine the range of the input intensities in the masked region
    let g = x.par_iter().zip(mask.par_iter()).filter(|(_,m)| {
        !m.is_zero()
    })
        .map(|(x,_)| *x)
        .collect::<Vec<f64>>();
    let max = *g.iter().max_by(|a,b| a.partial_cmp(b).expect("unable to compare values")).expect("no max value found");
    let min = *g.iter().min_by(|a,b| a.partial_cmp(b).expect("unable to compare values")).expect("no min value found");

    assert!(max.is_finite());
    assert!(min.is_finite());

    let range = max - min;

    // discretize x into bins
    d.par_iter_mut().zip(mask.par_iter().zip(x.par_iter())).for_each(|(d,(m,&x))| {
        if m.is_zero() {
            *d = 0;
            return;
        }
        *d = if x < max {
            ((n_bins as f64 * (x - min) / range).floor() + 1.) as u16
        }else {
            n_bins as u16
        };
    })

}

/// discretize gray level intensities 'x' into bins within some roi 'mask' given a 'bin_width.' The result is
/// written to 'd' as unsigned 16-bit values
fn discretize_bin_width<M:Zero + Copy + Send + Sync>(bin_width:f64, x:&[f64], mask:&[M], d:&mut [u16]) {

    assert_eq!(mask.len(), x.len());
    assert_eq!(d.len(), x.len());

    let min = x.par_iter().zip(mask.par_iter()).filter(|(_,m)| {
        !m.is_zero()
    })
        .map(|(x,_)| *x).min_by(|a,b|a.partial_cmp(b)
        .expect("unable to compare values"))
        .expect("no min value found");

    assert!(min.is_finite());

    // discretize x into bins
    d.iter_mut().zip(mask.iter().zip(x.iter())).for_each(|(d,(m,&x))| {
        if m.is_zero() {
            *d = 0;
            return;
        }
        let disc:f64 = (x / bin_width).floor() - (min / bin_width).floor() + 1.;
        *d = disc.to_u16().expect("number of bins exceeds u16::MAX with this bin width");
    })

}


/// returns the number of angles associated with some infinity norm 'r' in 3-D
pub fn n_angles(r:usize) -> usize {
    ((r*2 + 1).pow(3) - 1) / 2
}

/// generate a set of angles with some distance 'r' (infinity norm)
pub fn generate_angles(angles: &mut [[i32; 3]], r: usize) {
    assert!(r >= 1, "r must be positive");
    let n_expected =  n_angles(r);
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
                let first_non_zero = if x != 0 { x } else if y != 0 { y } else { z };
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