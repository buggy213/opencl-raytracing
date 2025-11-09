use std::f32;

use raytracing::{geometry::{Complex, Vec2, Vec3}, materials::Material};
use tracing::warn;

use crate::{sample::{self, sample_cosine_hemisphere}, texture::CpuTextures};

#[derive(Debug)]
pub(crate) struct BsdfSample {
    pub(crate) wi: Vec3,
    pub(crate) bsdf: Vec3,
    pub(crate) pdf: f32
}

// Corresponds to `Material` evaluated at a specific point
pub enum CpuBsdf {
    Diffuse { 
        albedo: Vec3
    },

    SmoothDielectric { 
        eta: f32 
    },
    SmoothConductor { 
        eta: Complex
    },
}

impl CpuBsdf {
    // local coordinates, wo and wi both in upper hemisphere
    pub(crate) fn evaluate_bsdf(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        match self {
            CpuBsdf::Diffuse { albedo } => {
                *albedo / f32::consts::PI
            },

            // these are perfectly specular materials, so their
            // BSDF is composed of delta functions; we consider
            // the exact direction impossible to sample through
            // normal means 
            CpuBsdf::SmoothDielectric { .. } => Vec3::zero(),
            CpuBsdf::SmoothConductor { .. } => Vec3::zero()
        }
    }

    pub(crate) fn sample_bsdf(&self, wo: Vec3) -> BsdfSample {
        match self {
            CpuBsdf::Diffuse { albedo } => {
                // cosine-weighted hemisphere sampling
                let (wi, pdf) = sample_cosine_hemisphere();
                BsdfSample {
                    wi,
                    bsdf: *albedo / f32::consts::PI,
                    pdf,
                }
            },

            CpuBsdf::SmoothDielectric { eta } => {
                let normal = Vec3(0.0, 0.0, 1.0);

                #[allow(non_snake_case, reason = "physics convention")]
                let R = fresnel_dielectric(wo.z(), *eta);
                #[allow(non_snake_case, reason = "physics convention")]
                let T = 1.0 - R;

                // we randomly choose to sample the Dirac in the reflected direction,
                // or in the refracted direction, proportional to calculated reflection
                // and transmission coefficients
                let sample = sample::sample_uniform();
                let bsdf_sample = if sample < R {
                    let reflection_dir = Vec3::reflect(wo, normal);
                    let f = R / wo.z();
                    let pdf = R;
                    
                    BsdfSample {
                        wi: reflection_dir,
                        bsdf: Vec3(f, f, f),
                        pdf,
                    }
                }
                else {
                    let refract_dir = refract(*eta, wo, normal);
                    let Some(refract_dir) = refract_dir else {
                        warn!("encountered total internal reflection after sampling 
                            refraction direction, this should not be possible");
                        
                        return BsdfSample {
                            wi: wo,
                            bsdf: Vec3::zero(),
                            pdf: 1.0,
                        }
                    };

                    // PBRT 4ed 9.5.2
                    // refraction compresses / expands directions, and since radiance is 
                    // expressed per solid angle, it needs to account for this, otherwise
                    // the transmission part of BSDF is not energy-preserving when you integrate
                    let f = (T / wo.z()) / (eta * eta);

                    let pdf = 1.0;
                    BsdfSample {
                        wi: refract_dir,
                        bsdf: Vec3(f, f, f),
                        pdf
                    }
                };

                bsdf_sample
            },
            CpuBsdf::SmoothConductor { eta } => {
                let normal = Vec3(0.0, 0.0, 1.0);
                let reflection_dir = Vec3::reflect(wo, normal);
                let f = fresnel_complex(wo.z(), *eta) / wo.z();
                // pdf is also a delta, to cancel with the implied delta function in the BSDF
                let pdf = 1.0;
                BsdfSample {
                    wi: reflection_dir,
                    bsdf: Vec3(f, f, f), // TODO: wavelength-dependence
                    pdf 
                }
            },
        }
    }

    // if BSDF only has deltas, then doing light sampling is not worthwhile
    pub(crate) fn is_delta_bsdf(&self) -> bool {
        match self {
            CpuBsdf::Diffuse { .. } => false,
            CpuBsdf::SmoothDielectric { .. } => true,
            CpuBsdf::SmoothConductor { .. } => true,
        }
    }
}

pub(crate) trait CpuMaterial {
    // evaluate the material at a specific shading point to get a bsdf
    fn get_bsdf(&self, uv: Vec2, textures: &CpuTextures) -> CpuBsdf;
}

impl CpuMaterial for Material {
    fn get_bsdf(&self, uv: Vec2, textures: &CpuTextures) -> CpuBsdf {
        match self {
            Material::Diffuse { albedo } => {
                let albedo = textures.sample(*albedo, uv.u(), uv.v());
                let albedo = Vec3(albedo.r(), albedo.g(), albedo.b());
                
                CpuBsdf::Diffuse { albedo }
            },
            Material::SmoothDielectric { eta } => {
                let eta = textures.sample(*eta, uv.u(), uv.v()).r();

                CpuBsdf::SmoothDielectric { eta }
            },
            Material::SmoothConductor { eta } => {
                let eta = textures.sample(*eta, uv.u(), uv.v());
                let eta = Complex(eta.0, eta.1);

                CpuBsdf::SmoothConductor { eta }
            },
        }
    }
}

fn refract(mut eta: f32, wo: Vec3, mut normal: Vec3) -> Option<Vec3> {
    let mut cos_theta_i = Vec3::dot(wo, normal);
    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
        normal = -normal;
    }

    let sin_theta_2_i = 1.0 - cos_theta_i * cos_theta_i;
    let sin_theta_2_t = sin_theta_2_i / (eta * eta); // snell's law
    if sin_theta_2_t >= 1.0 {
        // total internal reflection
        return None;
    } 

    let cos_theta_t = (1.0 - sin_theta_2_t).sqrt();
    Some(-wo / eta + (cos_theta_i / eta - cos_theta_t) * normal)
}

// fraction of reflected light from Fresnel equations
// TODO: wavelength-dependence?
fn fresnel_dielectric(mut cos_theta_i: f32, mut eta: f32) -> f32 {
    // if the incident ray is on the backside (as determined by normal), then 
    // transmission is from dielectric material to air. It's easiest to just 
    // flip eta and cos_theta
    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
    }

    let sin_theta_2_i = 1.0 - cos_theta_i * cos_theta_i;
    // Snell's law to get refracted angle
    let sin_theta_2_t = sin_theta_2_i / (eta * eta); 
    if sin_theta_2_t >= 1.0 {
        // total internal reflection
        return 1.0;
    }

    let cos_theta_t = (1.0 - sin_theta_2_t).sqrt();

    let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i - cos_theta_t);
    let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    (r_parl * r_parl + r_perp * r_perp) / 2.0
}

// fraction of reflected light from Fresnel equations
// TODO: wavelength-dependence?
fn fresnel_complex(cos_theta_i: f32, eta: Complex) -> f32 {
    // note 1: we don't flip eta / cos_theta_i here since extinction within conductor means that
    // no rays should be coming from back
    // note 2: technically, this function completely subsumes `fresnel_dielectric`, barring the flip detail
    // (it works when there's total internal reflection too; r_parl and r_perp both are just dividing a 
    //  number by its conjugate, so both have modulus 1)
    // working with real numbers is more efficient, though

    let sin_theta_2_i = 1.0 - cos_theta_i * cos_theta_i;
    let sin_theta_2_t: Complex = sin_theta_2_i / (eta * eta);
    let cos_theta_t: Complex = -sin_theta_2_t + 1.0;

    let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    (r_parl.square_magnitude() + r_perp.square_magnitude()) / 2.0
}