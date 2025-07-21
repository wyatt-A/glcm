mod glcm;
mod core;
mod subr;

#[cfg(test)]
mod tests {
    use array_lib::ArrayDim;
    use array_lib::io_nifti::{read_nifti, write_nifti, write_nifti_with_header};
    use rayon::prelude::*;
    use crate::glcm::map_glcm;
    use crate::subr::change_dims;

    #[test]
    fn test_map_accuracy() {

        let (mask,mask_dims,mask_h) = read_nifti::<f64>("B:/ProjectSpace/wa41/RadMapTest/24SD022/orthob2000/MASK.nii");

        let mask:Vec<_> = mask.par_iter().map(|&x| if x == 0. {0u8} else {1u8}).collect();

        let n_bins = 64;
        let n_features = 24;
        let patch_radius = 1;
        let angles = [
            [ 1 , 1 , 1],[ 1 , 1 , 0],[ 1 , 1 ,-1],[ 1 , 0 , 1],
            [ 1 , 0 , 0],[ 1,  0 ,-1], [ 1, -1 , 1], [ 1, -1 , 0],
            [ 1 ,-1 ,-1],[ 0 , 1 , 1],[ 0 , 1 , 0],[ 0 , 1 ,-1],
            [ 0,  0 , 1],
        ];

        let (mut img,img_dims,img_h) = read_nifti::<f64>("B:/ProjectSpace/wa41/RadMapTest/24SD022/orthob2000/Reg_2000_avg-ds.nii");

        // img.par_iter_mut().zip(mask.par_iter()).for_each(|(x, &m)|{
        //     if m == 0 {
        //         *x = 0.;
        //     }
        // });

        // descretize to 1..=n_bins
        let dims = &img_dims.shape()[0..3];
        let img_max = *img.par_iter().zip(mask.par_iter()).filter_map(|(x,m)| if *m > 0 {Some(x)} else {None}).max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
        let img_min = *img.par_iter().zip(mask.par_iter()).filter_map(|(x,m)| if *m > 0 {Some(x)} else {None}).min_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();

        assert!(img_max.is_finite());
        assert!(img_min.is_finite());

        println!("image max = {}",img_max);
        println!("image min = {}",img_min);

        let range = img_max - img_min;
        // set image between 0 and 1
        img.par_iter_mut().for_each(|x| *x = (*x - img_min) / range);
        // scale up to n_bins - 1 ... (0..63), then add 1 so the output is 1...=n_bins
        let mut img:Vec<_> = img.into_par_iter().map(|x| (x * (n_bins-1) as f64).round())
            .map(|x| x as u16 + 1)
            //.map(|x| x as u16)
            .collect();

        img.par_iter_mut().zip(mask.par_iter()).for_each(|(x, &m)|{
            if m == 0 {
                *x = 0;
            }
        });

        let feature_dims = ArrayDim::from_shape(&[n_features,dims[0],dims[1],dims[2]]);
        let out_dims = ArrayDim::from_shape(&[dims[0],dims[1],dims[2],n_features]);
        let mut features = feature_dims.alloc(0f64);

        println!("running glcm ...");
        map_glcm(dims,&img,n_features,&mut features,&angles,n_bins,patch_radius,&[]);
        println!("done.");

        let mut out = out_dims.alloc(0f64);

        change_dims(dims,n_features,&features,&mut out);

        println!("writing results ...");
        write_nifti_with_header("B:/ProjectSpace/wa41/RadMapTest/24SD022/orthob2000/features.nii",&out[0..img_dims.numel()],img_dims,&img_h);

    }


}