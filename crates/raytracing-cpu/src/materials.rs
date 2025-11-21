use std::f32;

use raytracing::{geometry::{Complex, Vec2, Vec3}, materials::Material};
use tracing::warn;

use crate::{sample, texture::CpuTextures};

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
        eta: Vec3,
        kappa: Vec3,
    },

    RoughConductor {
        eta: Vec3,
        kappa: Vec3,
        alpha_x: f32,
        alpha_y: f32,
    },
    RoughDielectric {
        eta: f32,
        alpha_x: f32,
        alpha_y: f32,
    }
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
            CpuBsdf::SmoothConductor { .. } => Vec3::zero(),
            CpuBsdf::RoughConductor { 
                eta, 
                kappa,
                alpha_x, 
                alpha_y 
            } => {
                microfacet::torrance_sparrow_refl_bsdf(
                    wo,
                    wi,
                    *eta,
                    *kappa,
                    *alpha_x,
                    *alpha_y
                )
            }
            CpuBsdf::RoughDielectric { 
                eta, 
                alpha_x, 
                alpha_y 
            } => {
                microfacet::torrance_sparrow_bsdf(
                    wo,
                    wi,
                    *eta,
                    *alpha_x,
                    *alpha_y
                )
            }
        }
    }

    pub(crate) fn sample_bsdf(&self, wo: Vec3) -> BsdfSample {
        match self {
            CpuBsdf::Diffuse { albedo } => {
                // cosine-weighted hemisphere sampling
                let (wi, pdf) = sample::sample_cosine_hemisphere();
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
                    let cos_theta = reflection_dir.z().abs();
                    let f = R / cos_theta;
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
                    
                    // eta might be inverted if interaction is behind surface
                    let eta = if wo.z() < 0.0 {
                        1.0 / *eta
                    } else {
                        *eta
                    };
                    let cos_theta = refract_dir.z().abs();
                    let f = (T / cos_theta) / (eta * eta);

                    let pdf = T;
                    BsdfSample {
                        wi: refract_dir,
                        bsdf: Vec3(f, f, f),
                        pdf
                    }
                };

                bsdf_sample
            },

            CpuBsdf::SmoothConductor { eta, kappa } => {
                let normal = Vec3(0.0, 0.0, 1.0);
                let reflection_dir = Vec3::reflect(wo, normal);
                let f_r = fresnel_complex(wo.z(), Complex(eta.0, kappa.0)) / wo.z();
                let f_g = fresnel_complex(wo.z(), Complex(eta.1, kappa.1)) / wo.z();
                let f_b = fresnel_complex(wo.z(), Complex(eta.2, kappa.2)) / wo.z();
                let f = Vec3(f_r, f_g, f_b);
                
                // pdf is also a delta, to cancel with the implied delta function in the BSDF
                let pdf = 1.0;
                BsdfSample {
                    wi: reflection_dir,
                    bsdf: f, // TODO: wavelength-dependence
                    pdf 
                }
            },

            CpuBsdf::RoughConductor { 
                eta, 
                kappa,
                alpha_x, 
                alpha_y 
            } => {
                microfacet::torrance_sparrow_refl_sample(
                    wo, 
                    *eta, 
                    *kappa,
                    *alpha_x, 
                    *alpha_y
                )
            },

            CpuBsdf::RoughDielectric { 
                eta, 
                alpha_x, 
                alpha_y 
            } => {
                microfacet::torrance_sparrow_sample(
                    wo, 
                    *eta, 
                    *alpha_x, 
                    *alpha_y
                )
            }
        }
    }

    // if BSDF only has deltas, then doing light sampling is not worthwhile
    pub(crate) fn is_delta_bsdf(&self) -> bool {
        match self {
            CpuBsdf::Diffuse { .. } => false,
            CpuBsdf::SmoothDielectric { .. } => true,
            CpuBsdf::SmoothConductor { .. } => true,
            CpuBsdf::RoughConductor { .. } => false,
            CpuBsdf::RoughDielectric { .. } => false,
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
            Material::SmoothConductor { eta, kappa } => {
                let eta = textures.sample(*eta, uv.u(), uv.v());
                let kappa = textures.sample(*kappa, uv.u(), uv.v());
                let eta = Vec3(eta.r(), eta.g(), eta.b());
                let kappa = Vec3(kappa.r(), kappa.g(), kappa.b());

                CpuBsdf::SmoothConductor { eta, kappa }
            },

            Material::RoughConductor { eta, kappa, roughness } => {
                let eta = textures.sample(*eta, uv.u(), uv.v());
                let kappa = textures.sample(*kappa, uv.u(), uv.v());
                let eta = Vec3(eta.0, eta.1, eta.2);
                let kappa = Vec3(kappa.0, kappa.1, kappa.2);

                let roughness = textures.sample(*roughness, uv.u(), uv.v());
                let alpha_x = roughness.0.sqrt();
                let alpha_y = roughness.1.sqrt();
                CpuBsdf::RoughConductor { 
                    eta, 
                    kappa,
                    alpha_x, 
                    alpha_y 
                }
            }

            Material::RoughDielectric { eta, roughness } => {
                let eta = textures.sample(*eta, uv.u(), uv.v()).r();

                let roughness = textures.sample(*roughness, uv.u(), uv.v());
                let alpha_x = roughness.0.sqrt();
                let alpha_y = roughness.1.sqrt();

                CpuBsdf::RoughDielectric { eta, alpha_x, alpha_y }
            }

            _ => todo!("support new material")
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

    let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
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

// Trowbridge-Reitz microfacet distribution helper functions
mod microfacet {
    use std::f32;

    use raytracing::geometry::{Complex, Vec2, Vec3};

    use crate::{materials::{BsdfSample, fresnel_complex, fresnel_dielectric}, sample};

    // PBRT 4ed 9.6.1
    // represents relative differential area of microfacets pointing in a particular direction
    // when microfacets distributed according to "ellipsoidal bumps"
    // as usual, wm is in shading coordinate frame (+z <=> normal of "macrosurface")
    pub(super) fn distribution(wm: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
        let cos_theta = wm.z();
        let cos_theta_2 = cos_theta * cos_theta;
        let sin_theta_2 = 1.0 - cos_theta_2;

        let cos_phi = wm.x();
        let sin_phi = wm.y();
        
        let e = (cos_phi * cos_phi) / (alpha_x * alpha_x) + (sin_phi * sin_phi) / (alpha_y * alpha_y);
        let t = (1.0 + (sin_theta_2 / cos_theta_2) * e).powi(2);

        1.0 / (f32::consts::PI * alpha_x * alpha_y * cos_theta_2 * cos_theta_2 * t)
    }

    // PBRT 4ed 9.6.2
    // part of "Smith's approximation" to masking function G1, which accounts for microfacets
    // blocking other microfacets
    pub(super) fn lambda(w: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
        let cos_theta = w.z();
        let cos_theta_2 = cos_theta * cos_theta;
        let sin_theta_2 = 1.0 - cos_theta_2;
        let tan_theta_2 = sin_theta_2 / cos_theta_2;

        let cos_phi = w.x();
        let sin_phi = w.y();

        let alpha_2 = alpha_x * alpha_x * cos_phi * cos_phi + alpha_y * alpha_y * sin_phi * sin_phi;
        ((1.0 + alpha_2 * tan_theta_2).sqrt() - 1.0) / 2.0
    }

    #[allow(non_snake_case, reason = "physics convention")]
    pub(super) fn G1(w: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
        1.0 / (1.0 + lambda(w, alpha_x, alpha_y))
    }

    // PBRT 4ed 9.6.3
    // an approximation to the masking-shadowing function (shadowing is masking but in the outgoing direction)
    #[allow(non_snake_case, reason = "physics convention")]
    pub(super) fn G(wo: Vec3, wi: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
        1.0 / (1.0 + lambda(wo, alpha_x, alpha_y) + lambda(wi, alpha_x, alpha_y))
    }

    // PBRT 4ed 9.6.4
    // distribution of normals which are visible from some direction
    // (leads to improved sampling efficiency)
    pub(super) fn visible_distribution(w: Vec3, wm: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
        let cos_theta = w.z().abs();

        (G1(w, alpha_x, alpha_y) / cos_theta) * 
        distribution(wm, alpha_x, alpha_y) * 
        Vec3::dot(w, wm).abs()
    }

    // PBRT 4ed 9.6.4
    // sample from visible normals distribution
    // figure 9.29 visually represents what this is doing
    pub(super) fn sample_wm(w: Vec3, alpha_x: f32, alpha_y: f32) -> Vec3 {
        let mut wh = Vec3(alpha_x * w.x(), alpha_y * w.y(), w.z()).unit();
        if wh.z() < 0.0 { wh = -wh; }

        let p = sample::sample_unit_disk();
        let t1 = if wh.z() < 0.9999 {
            Vec3::cross(Vec3(0.0, 0.0, 1.0), wh)
        }
        else {
            Vec3(1.0, 0.0, 0.0)
        };
        let t2 = Vec3::cross(wh, t1);

        let h = (1.0 - p.x() * p.x()).sqrt();
        
        // map [-h, h] to [-h cos(theta), h]
        let offset = 0.5 * h * (1.0 - wh.z());
        let scale = 0.5 * (1.0 + wh.z());
        let p = Vec2(p.0, offset + scale * p.1);

        // project warped disk to hemisphere
        let pz = f32::max(0.0, 1.0 - p.square_magnitude()).sqrt();
        let nh: Vec3 = p.x() * t1 + p.y() * t2 + pz * wh;
        
        // normals transform according to inverse transpose of transform
        // here, transform is diagonal, with 1/alpha_x and 1/alpha_y scaling in x / y directions
        // to undo the scaling by alpha_x, alpha_y at the beginning of function
        Vec3(alpha_x * nh.x(), alpha_y * nh.y(), f32::max(1.0e-6, nh.z()))
            .unit()
    }

    // PBRT 4ed 9.6.5
    // microfacet distribution is not enough to create a BRDF
    // torrance-sparrow model is a general BRDF which assumes specular reflection
    // off of a (general) microfacet distribution, which we specialize to trowbridge-reitz. 
    // does *not* account for refraction

    // pdf
    pub(super) fn torrance_sparrow_refl_pdf(wo: Vec3, wi: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
        // protect against grazing angle reflections
        if (wo + wi) == Vec3::zero() { return 0.0; } 
        let mut wm = (wo + wi).unit();
        if wm.z() < 0.0 { wm = -wm; }

        visible_distribution(wo, wm, alpha_x, alpha_y) / (4.0 * Vec3::dot(wo, wm).abs())
    }

    // more accurately a brdf
    pub(super) fn torrance_sparrow_refl_bsdf(
        wo: Vec3, 
        wi: Vec3,
        eta: Vec3,
        kappa: Vec3,
        alpha_x: f32, 
        alpha_y: f32, 
    ) -> Vec3 {
        // protect against grazing angle reflections
        if (wo + wi) == Vec3::zero() { return Vec3::zero(); }
        let wm = (wo + wi).unit();
        let cos_theta = Vec3::dot(wm, wi);
        let fresnel = Vec3(
            fresnel_complex(cos_theta.abs(), Complex(eta.0, kappa.0)),
            fresnel_complex(cos_theta.abs(), Complex(eta.1, kappa.1)),
            fresnel_complex(cos_theta.abs(), Complex(eta.2, kappa.2))
        );

        let brdf = distribution(wm, alpha_x, alpha_y)
        * fresnel
        * G(wo, wi, alpha_x, alpha_y)
        / (4.0 * wo.z() * wi.z());
        
        brdf
    }

    pub(super) fn torrance_sparrow_refl_sample(
        wo: Vec3,
        eta: Vec3,
        kappa: Vec3,
        alpha_x: f32,
        alpha_y: f32,
    ) -> BsdfSample {
        let wm = sample_wm(wo, alpha_x, alpha_y);
        let wi = Vec3::reflect(wo, wm);
        
        // pbrt notes that this leads to energy loss
        if wm.z() * wi.z() < 0.0 {
            return BsdfSample { wi, bsdf: Vec3::zero(), pdf: 1.0 };
        }

        let pdf = torrance_sparrow_refl_pdf(
            wo, 
            wi, 
            alpha_x, 
            alpha_y
        );

        let bsdf = torrance_sparrow_refl_bsdf(
            wo, 
            wi, 
            eta, 
            kappa,
            alpha_x, 
            alpha_y
        );

        BsdfSample { 
            wi, 
            bsdf, 
            pdf 
        }
    }

    // PBRT 4ed 9.7
    pub(super) fn torrance_sparrow_pdf(
        wo: Vec3,
        wi: Vec3,
        eta: f32,
        alpha_x: f32,
        alpha_y: f32,
    ) -> f32 {
        let reflect = wo.z() * wi.z() > 0.0;
        let eta_wm = if !reflect {
            if wo.z() > 0.0 { eta } else { 1.0 / eta }
        } else { 
            1.0 
        };
        
        // "generalized half-direction vector" transform, wm must always face "up" in 
        // local coordinate frame
        let wm = wi * eta_wm + wo;
        let mut wm = wm.unit();

        if wm.z() < 0.0 {
            wm = -wm;
        }

        // guard against grazing angles
        if wi.z() == 0.0 || wo.z() == 0.0 || wm == Vec3::zero() {
            return 0.0;
        }

        // why is this correct? it "discards backfacing microfacets"
        // seems like it should be a little trickier than this
        if Vec3::dot(wm, wi) * wi.z() < 0.0 || Vec3::dot(wm, wo) * wo.z() < 0.0 {
            return 0.0;
        }
        
        // we use absolute value and eta_wm to avoid fresnel_dielectric flipping the interface again
        #[allow(non_snake_case, reason = "physics convention")]
        let R = fresnel_dielectric(
            Vec3::dot(wo, wm).abs(), 
            eta_wm
        );
        #[allow(non_snake_case, reason = "physics convention")]
        let T = 1.0 - R;

        if reflect {
            R * visible_distribution(wo, wm, alpha_x, alpha_y) / (4.0 * Vec3::dot(wo, wm).abs())
        }
        else {
            let denom = (Vec3::dot(wi, wm) + Vec3::dot(wo, wm) / eta_wm).powi(2);
            let dwm_dwi = Vec3::dot(wi, wm).abs() / denom;
            T * visible_distribution(wo, wm, alpha_x, alpha_y) * dwm_dwi
        }
    }

    pub(super) fn torrance_sparrow_bsdf(
        wo: Vec3,
        wi: Vec3,
        eta: f32,
        alpha_x: f32,
        alpha_y: f32
    ) -> Vec3 {
        let reflect = wo.z() * wi.z() > 0.0;
        let eta_wm = if !reflect {
            if wo.z() > 0.0 { eta } else { 1.0 / eta }
        } else { 
            1.0 
        };
        
        // "generalized half-direction vector" transform, wm must always face "up" in 
        // local coordinate frame
        let wm = wi * eta_wm + wo;
        let mut wm = wm.unit();

        if wm.z() < 0.0 {
            wm = -wm;
        }

        // guard against grazing angles
        if wi.z() == 0.0 || wo.z() == 0.0 || wm == Vec3::zero() {
            return Vec3::zero();
        }

        // we use absolute value and eta_wm to avoid fresnel_dielectric flipping the interface again
        #[allow(non_snake_case, reason = "physics convention")]
        let F = fresnel_dielectric(
            Vec3::dot(wo, wm).abs(), 
            eta_wm
        );

        if reflect {
            let brdf = distribution(wm, alpha_x, alpha_y)
            * F
            * G(wo, wi, alpha_x, alpha_y)
            / (4.0 * wo.z() * wi.z()).abs();

            Vec3(brdf, brdf, brdf)
        }
        else {
            let denom = wi.z() * wo.z() * (Vec3::dot(wi, wm) + Vec3::dot(wo, wm) / eta_wm).powi(2);
            let btdf = distribution(wm, alpha_x, alpha_y)
            * (1.0 - F)
            * G(wo, wi, alpha_x, alpha_y)
            * (Vec3::dot(wi, wm) * Vec3::dot(wo, wm) / denom).abs()
            / (eta_wm * eta_wm);

            Vec3(btdf, btdf, btdf)
        }
    }

    pub(super) fn torrance_sparrow_sample(
        wo: Vec3,
        eta: f32,
        alpha_x: f32,
        alpha_y: f32,
    ) -> BsdfSample {
        let wm = sample_wm(wo, alpha_x, alpha_y);
         
        #[allow(non_snake_case, reason = "physics convention")]
        let R = fresnel_dielectric(Vec3::dot(wo, wm), eta);
        #[allow(non_snake_case, reason = "physics convention")]
        let _T = 1.0 - R;

        let wi = if sample::sample_uniform() < R {
            let wi = Vec3::reflect(wo, wm);
            if wo.z() * wi.z() < 0.0 {
                return BsdfSample { wi, bsdf: Vec3::zero(), pdf: 1.0 }
            }

            wi
        }
        else {
            let wi = super::refract(
                eta, 
                wo, 
                wm
            );

            let Some(wi) = wi else {
                return BsdfSample { wi: Vec3(0.0, 0.0, 1.0), bsdf: Vec3::zero(), pdf: 1.0 }
            };

            if wo.z() * wi.z() > 0.0 || wi.z() == 0.0 {
                return BsdfSample { wi, bsdf: Vec3::zero(), pdf: 1.0 }
            }

            wi
        };

        let pdf = torrance_sparrow_pdf(
            wo, 
            wi, 
            eta, 
            alpha_x, 
            alpha_y
        );

        let bsdf = torrance_sparrow_bsdf(
            wo,
            wi,
            eta,
            alpha_x,
            alpha_y
        );

        BsdfSample {
            wi,
            bsdf,
            pdf,
        }
    }
}


mod gltf_pbr {
    // the gltf metallic-roughness material essentially 
    // describes a parametetric mix material between diffuse and 
    // specular components, so we can evaluate it and sample it as such

    // GLTF spec B.3.5
}