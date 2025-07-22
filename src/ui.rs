use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::{Display, Formatter, Write};
use std::str::FromStr;
use serde::{Deserialize, Serialize};
use strum::EnumDiscriminants;

#[cfg(test)]
mod tests {
    use crate::ui::{Alias, GLCMFeature, RadMapOpts, RadMapOptsSer};

    #[test]
    fn config() {

        let opts = RadMapOptsSer {
            features: vec![
                Alias { feature: "auto_correlation".to_string(), alias: "autocorr".to_string() },
                Alias { feature: "auto_correlation".to_string(), alias: "autocorr".to_string() },
            ],
            separator: Some("_".to_string()),
        };

        let ts = toml::to_string(&opts).unwrap();
        println!("{}", ts);

        let opts:RadMapOpts = opts.into();

        println!("{:?}", opts);

        println!("{}",GLCMFeature::CONTRAST.index());

        let ops:RadMapOptsSer = RadMapOpts::default().into();
        let ts = toml::to_string(&ops).unwrap();
        println!("{ts}")

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
            "INVERSE_DIFFERENCE_MOMENT_NORMALIZED" => {
                Ok(INVERSE_DIFFERENCE_MOMENT_NORMALIZED)
            }
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



#[derive(Debug)]
pub struct RadMapOpts {
    pub separator: Option<String>,
    pub features: HashMap<GLCMFeature,String>,
}

use strum::{Display, EnumIter, IntoEnumIterator};

impl RadMapOpts {
    pub fn default() -> Self {
        let separator = "_".to_string();
        let features = GLCMFeature::iter()
            .map(|feat| (feat, feat.to_string().to_lowercase()))
            .collect();

        RadMapOpts {
            separator: Some(separator),
            features,
        }
    }

    pub fn separator(&self) -> &str {
        self.separator.as_deref().unwrap_or("_")
    }

    pub fn features(&self) -> Vec<(GLCMFeature,String)> {
        let mut f:Vec<_> = self.features.iter().map(|(k,v)| (*k,v.clone())).collect();
        f.sort_by_key(|k|k.1.clone());
        f
    }

    pub fn contains(&self, feature: GLCMFeature) -> bool {
        self.features.contains_key(&feature)
    }

}


#[derive(Serialize,Deserialize,Debug)]
struct Alias {
    feature: String,
    alias: String,
}

#[derive(Serialize,Deserialize,Debug)]
pub struct RadMapOptsSer {
    features: Vec<Alias>,
    separator: Option<String>,
}

impl Into<RadMapOpts> for RadMapOptsSer {
    fn into(self) -> RadMapOpts {
        let mut h = HashMap::new();
        for alias in self.features {
            let f = GLCMFeature::from_str(&alias.feature)
                .expect(&format!("invalid glcm feature identifier: {}", alias.feature).as_str());
            h.insert(f, alias.alias);
        }
        RadMapOpts {
            separator: self.separator,
            features: h,
        }
    }
}

impl Into<RadMapOptsSer> for RadMapOpts {
    fn into(self) -> RadMapOptsSer {

        let mut f:Vec<_> = self.features.iter().map(|(k,v)|{
            Alias {
                feature: k.to_string(),
                alias: v.to_string(),
            }
        }).collect();

        f.sort_by_key(|x| x.alias.clone());

        RadMapOptsSer {
            features: f,
            separator: self.separator,
        }

    }
}