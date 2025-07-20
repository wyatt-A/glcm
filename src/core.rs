use array_lib::ArrayDim;
use rayon::prelude::*;

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use array_lib::ArrayDim;
    use rand::{Rng};
    use crate::subr::symmetrize_in_place;
    use super::{compute_glcm, n_neighbors};

    #[test]
    fn textbook_asymm() {

        // image patch dimensions
        let dims = [3,3,3];
        // r-vector that determines direction and distance
        let r = [1,0,0];
        //number of gray scale levels
        let n_levels = 4;

        // 3-D image patch from textbook example
        let mut x = ArrayDim::from_shape(&dims).alloc(0u16);
        x.copy_from_slice(
            &[0,0,1,0,1,2,0,2,3,1,2,3,0,2,3,0,1,2,1,3,0,0,3,1,3,2,1]
        );

        let mut addr_buff = vec![[0usize;2];n_neighbors(&dims,&r)];
        let mut glcm = vec![0usize;n_levels*n_levels];

        let now = Instant::now();
        compute_glcm(&dims,&x,&r,n_levels,&mut addr_buff,&mut glcm);
        let dur = now.elapsed();
        println!("Elapsed: {:.2?} us", dur.as_micros());

        let expected_glcm = [1,0,0,1,3,0,1,1,2,3,0,1,1,1,3,0];
        assert_eq!(&glcm,&expected_glcm);
    }

    /// test example from pyradiomics (symmetrical GLCM from 5x5 patch)
    #[test]
    fn textbook_symm() {

        // image patch dimensions
        let dims = [5,5,1];
        // r-vector that determines direction and distance
        let r = [0,1,0];
        //number of grayscale levels
        let n_levels = 5;

        let mut x = ArrayDim::from_shape(&dims).alloc(0u16);
        x.copy_from_slice(
            &[1,3,1,1,1,2,2,3,1,2,5,1,5,1,4,2,3,5,1,3,3,1,2,2,5]
        );
        x.iter_mut().for_each(|x| *x -= 1); // example data starts at 1, but we expect to start at 0

        let mut addr_buff = vec![[0usize;2];n_neighbors(&dims,&r)];
        let mut glcm = vec![0usize;n_levels*n_levels];

        let now = Instant::now();
        compute_glcm(&dims,&x,&r,n_levels,&mut addr_buff,&mut glcm);
        symmetrize_in_place(&mut glcm,n_levels);
        let dur = now.elapsed();
        println!("Elapsed: {:.2?} us", dur.as_micros());

        let expected_symm_glcm = [6,4,3,0,0,4,0,2,1,3,3,2,0,1,2,0,1,1,0,0,0,3,2,0,2];
        assert_eq!(&glcm,&expected_symm_glcm);

    }

    #[test]
    fn dim3_symm() {

        // image patch dimensions
        let dims = [3,3,3];
        // r-vector that determines direction and distance
        let r = [0,0,1];
        //number of grayscale levels
        let n_levels = 5;

        let mut x = ArrayDim::from_shape(&dims).alloc(0u16);
        x.copy_from_slice(
            &[1,1,2,2,4,5,3,4,2,3,2,1,4,2,5,3,1,1,5,3,2,1,1,5,4,4,1]
        );
        x.iter_mut().for_each(|x| *x -= 1); // example data starts at 1, but we expect to start at 0

        let mut addr_buff = vec![[0usize;2];n_neighbors(&dims,&r)];
        let mut glcm = vec![0usize;n_levels*n_levels];
        //let mut glcm_symm = vec![0usize;n_levels*n_levels];

        let now = Instant::now();
        compute_glcm(&dims,&x,&r,n_levels,&mut addr_buff,&mut glcm);
        symmetrize_in_place(&mut glcm,n_levels);
        //compute_symmetrical(n_levels,&glcm,&mut glcm_symm);
        let dur = now.elapsed();
        println!("Elapsed: {:.2?} us", dur.as_micros());

        println!("glcm: {:?}", glcm);
        //0, 3, 1, 1, 1, 3, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 1, 0, 0, 0

    }




    #[test]
    fn speed() {

        let dims = [9,9,9];
        let r = [1,0,0];
        let n_levels = 8;

        // generate a large random image patch
        let mut rng = rand::rng();
        let adim_x = ArrayDim::from_shape(&dims);
        let x:Vec<u16> = (0..adim_x.numel()).map(|_| rng.random_range(0..n_levels as u16)).collect();

        let mut addr_buff = vec![[0usize;2];n_neighbors(&dims,&r)];
        let mut glcm = vec![0usize;n_levels*n_levels];

        let now = Instant::now();
        compute_glcm(&dims,&x,&r,n_levels,&mut addr_buff,&mut glcm);
        let dur = now.elapsed();
        println!("Elapsed: {:.2?} us", dur.as_micros());

        println!("{:?}",glcm);

    }

}

/// symmetrical GLCM from asymmetrical GLCM (S = A^T + A) ... This can be done in-place (I think)
fn compute_symmetrical(n_levels:usize,asymm_glcm:&[usize],symm_glcm:&mut [usize]) {
    debug_assert_eq!(n_levels*n_levels,asymm_glcm.len());
    debug_assert_eq!(asymm_glcm.len(),symm_glcm.len());
    let glcm_dim = ArrayDim::from_shape(&[n_levels;2]);
    symm_glcm.par_iter_mut().zip(asymm_glcm.par_iter()).enumerate().for_each(|(idx,(s,&a))|{
        let [x,y,..] = glcm_dim.calc_idx(idx);
        let addr = glcm_dim.calc_addr(&[y,x]); // G^T address
        // calc A^T + A
        *s = asymm_glcm[addr] + a;
    })
}

fn compute_glcm(dims:&[usize], x:&[u16], r:&[usize], n_levels:usize, addr_buffer:&mut [[usize;2]], glcm:&mut [usize]) {

    // make sure buffer sizes and parameters make sense
    debug_assert!(n_levels*n_levels == glcm.len());
    debug_assert_eq!(dims.len(), 3);
    debug_assert_eq!(r.len(), 3);
    debug_assert!(x.len() == dims.iter().product());
    debug_assert!((n_levels - 1) <= u16::MAX as usize);
    debug_assert_eq!(n_neighbors(dims, r), addr_buffer.len());

    // dimension helpers to convert between element addresses and subscripts
    let adim_x = ArrayDim::from_shape(dims);
    let adim_glcm = ArrayDim::from_shape(&[n_levels;2]);

    // compute the address pairs (linear with size of x)
    let mut ii = 0;
    for i in 0..(dims[0] - r[0]) {
        for j in 0..(dims[1] - r[1]) {
            for k in 0..(dims[2] - r[2]) {
                let root_idx = [i,j,k];
                let neighbor_idx = [i + r[0], j + r[1], k + r[2]];
                let root_idx = adim_x.calc_addr(&root_idx);
                let neighbor_idx = adim_x.calc_addr(&neighbor_idx);
                addr_buffer[ii][0] = root_idx;
                addr_buffer[ii][1] = neighbor_idx;
                ii += 1;
            }
        }
    }

    // compute glcm (quadratic with n_levels)
    glcm.par_iter_mut().enumerate().for_each(|(addr,entry)| {
        let [i,j,..] = adim_glcm.calc_idx(addr);
        *entry = addr_buffer.par_iter().filter(|pair| {
            x[pair[0]] == i as u16 && x[pair[1]] == j as u16
        }).count();
    });
}

// this needs to be tested to support negative values in r
fn n_neighbors(dims:&[usize], r:&[usize]) -> usize {
    dims.iter().zip(r.iter()).for_each(|(d,r)|{
        assert!(r < d)
    });
    dims.iter().zip(r.iter()).map(|(&d,&r)| d - r).product::<usize>()
}