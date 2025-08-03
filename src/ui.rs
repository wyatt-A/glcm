use glcm::GLCMFeature;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::str::FromStr;
use std::{fmt, io};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseGLCMFeatureError;

impl Display for ParseGLCMFeatureError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("invalid GLCM feature name")
    }
}

impl std::error::Error for ParseGLCMFeatureError {}

impl FromStr for GLCMFeature {
    type Err = ParseGLCMFeatureError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use GLCMFeature::*;
        match s.to_ascii_uppercase().as_str() {
            "AUTO_CORRELATION" | "AUTOCORRELATION" => Ok(AUTO_CORRELATION),
            "JOINT_AVERAGE" => Ok(JOINT_AVERAGE),
            "CLUSTER_PROMINENCE" => Ok(CLUSTER_PROMINENCE),
            "CLUSTER_SHADE" => Ok(CLUSTER_SHADE),
            "CLUSTER_TENDENCY" => Ok(CLUSTER_TENDENCY),
            "CONTRAST" => Ok(CONTRAST),
            "CORRELATION" => Ok(CORRELATION),
            "DIFFERENCE_AVERAGE" => Ok(DIFFERENCE_AVERAGE),
            "DIFFERENCE_ENTROPY" => Ok(DIFFERENCE_ENTROPY),
            "DIFFERENCE_VARIANCE" => Ok(DIFFERENCE_VARIANCE),
            "JOINT_ENERGY" => Ok(JOINT_ENERGY),
            "JOINT_ENTROPY" => Ok(JOINT_ENTROPY),
            "IMC1" => Ok(IMC1),
            "IMC2" => Ok(IMC2),
            "INVERSE_DIFFERENCE_MOMENT" => Ok(INVERSE_DIFFERENCE_MOMENT),
            "MAXIMUM_CORRELATION_COEFFICIENT" => Ok(MAXIMUM_CORRELATION_COEFFICIENT),
            "INVERSE_DIFFERENCE_MOMENT_NORMALIZED" => Ok(INVERSE_DIFFERENCE_MOMENT_NORMALIZED),
            "INVERSE_DIFFERENCE" => Ok(INVERSE_DIFFERENCE),
            "INVERSE_DIFFERENCE_NORMALIZED" => Ok(INVERSE_DIFFERENCE_NORMALIZED),
            "INVERSE_VARIANCE" => Ok(INVERSE_VARIANCE),
            "MAXIMUM_PROBABILITY" => Ok(MAXIMUM_PROBABILITY),
            "SUM_AVERAGE" => Ok(SUM_AVERAGE),
            "SUM_ENTROPY" => Ok(SUM_ENTROPY),
            "SUM_OF_SQUARES" => Ok(SUM_OF_SQUARES),
            _ => Err(ParseGLCMFeatureError),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MapOpts {
    pub kernel_radius: usize,
    pub n_bins: usize,
    pub features: HashMap<GLCMFeature, String>,
    pub separator: Option<String>,
    pub max_threads: Option<usize>,
}

impl Default for MapOpts {
    fn default() -> Self {
        let separator = "_".to_string();
        let features = GLCMFeature::iter()
            .map(|feat| (feat, feat.to_string().to_lowercase()))
            .collect();

        MapOpts {
            kernel_radius: 1,
            separator: Some(separator),
            features,
            n_bins: 32,
            max_threads: None,
        }
    }
}

use crate::glcm;
use strum::IntoEnumIterator;

impl MapOpts {
    pub fn from_file(toml_file: impl AsRef<Path>) -> Result<Self, io::Error> {
        let mut f = File::open(toml_file.as_ref())?;
        let mut toml_str = String::new();
        f.read_to_string(&mut toml_str)?;
        let opts_ser: RadMapOptsSer =
            toml::from_str(&toml_str).map_err(|_| io::Error::from(io::ErrorKind::InvalidData))?;
        let opts: MapOpts = opts_ser.into();
        Ok(opts)
    }

    pub fn to_file(&self, toml_file: impl AsRef<Path>) -> Result<(), io::Error> {
        let mut f = File::create(toml_file.as_ref())?;
        let ser: RadMapOptsSer = self.clone().into();
        let ts = toml::to_string_pretty(&ser)
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidData))?;
        f.write_all(ts.as_bytes())?;
        Ok(())
    }

    pub fn separator(&self) -> &str {
        self.separator.as_deref().unwrap_or("_")
    }

    pub fn features(&self) -> HashSet<GLCMFeature> {
        let mut features = HashSet::new();
        self.features.iter().for_each(|(k, _)| {
            features.insert(*k);
        });
        features
    }

    pub fn features_aliases(&self) -> Vec<(GLCMFeature, String)> {
        let mut f: Vec<_> = self.features.iter().map(|(k, v)| (*k, v.clone())).collect();
        f.sort_by_key(|k| k.1.clone());
        f
    }

    pub fn contains(&self, feature: GLCMFeature) -> bool {
        self.features.contains_key(&feature)
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct Alias {
    feature: String,
    alias: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RadMapOptsSer {
    kernel_radius: i32,
    n_bins: usize,
    features: Vec<Alias>,
    separator: Option<String>,
}

impl From<RadMapOptsSer> for MapOpts {
    fn from(val: RadMapOptsSer) -> Self {
        let mut h = HashMap::new();
        for alias in val.features {
            let f = GLCMFeature::from_str(&alias.feature)
                .unwrap_or_else(|_| panic!("invalid glcm feature identifier: {}", alias.feature));
            h.insert(f, alias.alias);
        }
        MapOpts {
            kernel_radius: val.kernel_radius.unsigned_abs() as usize,
            separator: val.separator,
            features: h,
            n_bins: val.n_bins,
            max_threads: None,
        }
    }
}

impl From<MapOpts> for RadMapOptsSer {
    fn from(val: MapOpts) -> Self {
        let mut f: Vec<_> = val
            .features
            .iter()
            .map(|(k, v)| Alias {
                feature: k.to_string(),
                alias: v.to_string(),
            })
            .collect();

        f.sort_by_key(|x| x.alias.clone());

        RadMapOptsSer {
            kernel_radius: val.kernel_radius as i32,
            n_bins: val.n_bins,
            features: f,
            separator: val.separator,
        }
    }
}
