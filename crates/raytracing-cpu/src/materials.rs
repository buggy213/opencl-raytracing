use std::f32;

use raytracing::{geometry::{Complex, Vec2, Vec3}, materials::Material};
use tracing::warn;

use crate::{sample, texture::CpuTextures};

#[derive(Debug)]
pub(crate) struct BsdfSample {
    pub(crate) wi: Vec3,
    pub(crate) bsdf: Vec3,
    pub(crate) pdf: f32,

    // tracks whether this sample represents a specular component of 
    // bsdf; only gltf material returns both true / false at the moment
    pub(crate) specular: bool,
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
    },

    GLTFMetallicRoughness {
        base_color: Vec3,
        metallic: f32,
        alpha: f32,
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
            },
            CpuBsdf::GLTFMetallicRoughness { 
                base_color, 
                metallic, 
                alpha 
            } => {
                gltf_pbr::bsdf(
                    wo, 
                    wi, 
                    *base_color, 
                    *metallic, 
                    *alpha
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
                    specular: false,
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
                        specular: true,
                    }
                }
                else {
                    let refract_dir = refract(*eta, wo, normal);
                    let Some(refract_dir) = refract_dir else {
                        warn!(
                            "encountered total internal reflection after sampling \
                            refraction direction, this should not be possible as \
                            fresnel_dielectric should return 1.0"
                        );
                        
                        return BsdfSample {
                            wi: wo,
                            bsdf: Vec3::zero(),
                            pdf: 1.0,
                            specular: true,
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
                        pdf,
                        specular: true,
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
                    bsdf: f,
                    pdf,
                    specular: true
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
            },

            CpuBsdf::GLTFMetallicRoughness { 
                base_color, 
                metallic, 
                alpha 
            } => {
                gltf_pbr::sample_f(
                    wo, 
                    *base_color, 
                    *metallic, 
                    *alpha
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
            CpuBsdf::GLTFMetallicRoughness { .. } => false,
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

            Material::GLTFMetallicRoughness { base_color, metallic_roughness } => {
                let base_color = textures.sample(*base_color, uv.u(), uv.v());
                let base_color = Vec3(base_color.0, base_color.1, base_color.2);

                let metallic_roughness = textures.sample(*metallic_roughness, uv.u(), uv.v());
                let metallic = metallic_roughness.b().clamp(0.0, 1.0);
                let roughness = metallic_roughness.g().clamp(0.0, 1.0);
                let alpha = roughness * roughness;
                
                CpuBsdf::GLTFMetallicRoughness { 
                    base_color, 
                    metallic, 
                    alpha
                }
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
    let cos_theta_2_t: Complex = -sin_theta_2_t + 1.0;

    // not sure if taking principal branch is equivalent to other branch
    // of complex square root, but it's *probably* correct
    let cos_theta_t: Complex = cos_theta_2_t.sqrt();

    let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    (r_parl.square_magnitude() + r_perp.square_magnitude()) / 2.0
}

// Trowbridge-Reitz microfacet distribution helper functions
mod microfacet {
    use std::f32;

    use raytracing::geometry::{Complex, Vec2, Vec3};
    use tracing::warn;

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
        
        // this represents the case that the ray bounces deeper into
        // the microsurface; we discard these samples because we only
        // account for first-order scattering events in microfacet model
        // pbrt notes that this leads to energy loss and is thus not fully physical
        if wo.z() * wi.z() < 0.0 {
            return BsdfSample { wi, bsdf: Vec3::zero(), pdf: 1.0, specular: false };
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
            pdf,
            specular: false
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

        // wm is always in the upper hemisphere
        // if wi is in the upper hemisphere, it's backfacing if wm ⋅ wi < 0
        // if it's in the lower hemisphere, then it's backfacing if -wm ⋅ wi < 0
        // this condition discards backfacing microfacets, because we would never sample them
        if Vec3::dot(wm, wi) * wi.z() < 0.0 || Vec3::dot(wm, wo) * wo.z() < 0.0 {
            return 0.0;
        }
        
        #[allow(non_snake_case, reason = "physics convention")]
        let R = fresnel_dielectric(
            Vec3::dot(wo, wm), 
            eta
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

        // also need to discard backfacing microfacets here 
        if Vec3::dot(wm, wi) * wi.z() < 0.0 || Vec3::dot(wm, wo) * wo.z() < 0.0 {
            return Vec3::zero();
        }

        #[allow(non_snake_case, reason = "physics convention")]
        let F = fresnel_dielectric(
            Vec3::dot(wo, wm), 
            eta
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

            // see analogous comment in `torrance_sparrow_refl_sample`
            if wo.z() * wi.z() < 0.0 {
                return BsdfSample { wi, bsdf: Vec3::zero(), pdf: 1.0, specular: false, }
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
                warn!(
                    "encountered total internal reflection after sampling \
                    refraction direction, this should not be possible as \
                    fresnel_dielectric should return 1.0"
                );

                return BsdfSample { wi: Vec3(0.0, 0.0, 1.0), bsdf: Vec3::zero(), pdf: 1.0, specular: false, }
            };

            if wo.z() * wi.z() > 0.0 || wi.z() == 0.0 {
                warn!(
                    "weird floating point things happened; wi doesn't represent \
                    a valid transmission direction"
                );

                return BsdfSample { wi, bsdf: Vec3::zero(), pdf: 1.0, specular: false, }
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
            specular: false,
        }
    }
}


mod gltf_pbr {
    // the gltf metallic-roughness material essentially 
    // describes a parametetric mix material between diffuse and 
    // specular components.

    // note: the spec notes that their sample implementation 
    // violates energy conservation and reciprocity; thus using it
    // is actually in violation of GLTF spec (since it requires offline
    // physically-based renderers, which this project certainly is, to
    // have a physically plausible bsdf). however, they also leave
    // what the correct behavior is totally unspecified so...

    // roughness near 0 is treated as a special case to avoid
    // numerical problems 
    use std::f32;

    use raytracing::geometry::Vec3;

    use crate::{materials::BsdfSample, sample};

    #[allow(non_snake_case, reason = "gltf spec")]
    fn D(wm: Vec3, alpha: f32) -> f32 {
        super::microfacet::distribution(wm, alpha, alpha)
    }

    #[allow(non_snake_case, reason = "gltf spec")]
    fn G(wo: Vec3, wi: Vec3, alpha: f32) -> f32 {
        super::microfacet::G(wo, wi, alpha, alpha)
    }

    fn specular_brdf(alpha: f32, wo: Vec3, wi: Vec3) -> f32 {
        let half = if wo + wi == Vec3::zero() {
            return 0.0;
        }
        else {
            (wo + wi).unit()
        };

        D(half, alpha) * G(wo, wi, alpha) / (4.0 * wo.z().abs() * wi.z().abs())
    }

    fn conductor_fresnel(f0: Vec3, bsdf: Vec3, cos_theta: f32) -> Vec3 {
        bsdf * (f0 + (Vec3(1.0, 1.0, 1.0) - f0) * (1.0 - cos_theta).powi(5))
    }

    fn fresnel_mix_factor(ior: f32, cos_theta: f32) -> f32 {
        let f0 = ((1.0 - ior) / (1.0 + ior)).powi(2);
        let fr = f0 + (1.0 - f0) * (1.0 - cos_theta).powi(5);
        
        fr
    }

    fn fresnel_mix(ior: f32, base: Vec3, layer: Vec3, cos_theta: f32) -> Vec3 {
        let f = fresnel_mix_factor(ior, cos_theta);
        f * layer + (1.0 - f) * base
    }

    pub(super) fn pdf(
        wo: Vec3,
        wi: Vec3,
        metallic: f32,
        alpha: f32,
    ) -> f32 {
        if alpha < 1.0e-4 {
            // the material is effectively smooth, so the specular component is a delta
            // thus, we only need to consider the diffuse component, since it should be impossible
            // to call pdf with exactly the mirror direction of wo
            // (sample_f needs to ensure it handles effectively smooth case also, or else return wrong pdf)
            let dielectric_fresnel_mix = fresnel_mix_factor(1.5, wo.z());
            let specular_prob = metallic + (1.0 - metallic) * dielectric_fresnel_mix;
            let diffuse_prob = 1.0 - specular_prob;

            let diffuse_pdf = wi.z();

            return diffuse_prob * diffuse_pdf;
        }

        let wm = if wo + wi == Vec3::zero() {
            return 0.0;
        } else {
            (wo + wi).unit()
        };
        
        let dielectric_fresnel_mix = fresnel_mix_factor(1.5, Vec3::dot(wo, wm));

        // there are two 2 cases, corresponding to specular_prob and diffuse_prob 
        // 1. we sampled wi as a result of dielectric reflection or metallic reflection
        // 2. we sampled wi as a result of diffuse - this can only happen if z > 0
        let specular_prob = metallic + (1.0 - metallic) * dielectric_fresnel_mix;
        let diffuse_prob = 1.0 - specular_prob;

        let specular_pdf = super::microfacet::visible_distribution(wo, wm, alpha, alpha) / (4.0 * Vec3::dot(wo, wm).abs());
        let diffuse_pdf = wi.z();

        specular_prob * specular_pdf + diffuse_prob * diffuse_pdf
    }

    // direct transcription of GLTF spec B.3.5
    pub(super) fn bsdf(
        wo: Vec3,
        wi: Vec3,
        base_color: Vec3,
        metallic: f32,
        alpha: f32,
    ) -> Vec3 {
        if alpha < 1.0e-4 {
            // effectively smooth case; see comment in `gltf_pbr::pdf`
            let dielectric_fresnel_mix = fresnel_mix_factor(1.5, wo.z());
            let diffuse_factor = (1.0 - metallic) * (1.0 - dielectric_fresnel_mix);
            return diffuse_factor * base_color * f32::consts::FRAC_1_PI;
        }

        let c_diff = metallic * Vec3::zero() + (1.0 - metallic) * base_color;
        let f0 = metallic * base_color + (1.0 - metallic) * Vec3(0.04, 0.04, 0.04);
        
        let half = if wo + wi == Vec3::zero() {
            return Vec3::zero();
        } else {
            (wo + wi).unit()
        };

        let fresnel = f0 + (Vec3(1.0, 1.0, 1.0) - f0) * (1.0 - Vec3::dot(wo, half).abs()).powi(5);
        let f_diffuse = (Vec3(1.0, 1.0, 1.0) - fresnel) * (f32::consts::FRAC_1_PI) * c_diff;
        let f_specular = fresnel * specular_brdf(alpha, wo, wi);

        f_diffuse + f_specular
    }

    pub(super) fn sample_f(
        wo: Vec3,
        base_color: Vec3,
        metallic: f32,
        alpha: f32,
    ) -> BsdfSample {
        if alpha < 1.0e-4 {
            // effectively smooth case
            // we need to stochastically return the specular component or the diffuse component
            let dielectric_fresnel_mix = fresnel_mix_factor(1.5, wo.z());
            let specular_prob = metallic + (1.0 - metallic) * dielectric_fresnel_mix;
            
            if sample::sample_uniform() < specular_prob {
                // specular_brdf reduces to 1.0 across all wavelengths in the case that it's fully smooth
                let metal_component = metallic * conductor_fresnel(base_color, Vec3(1.0, 1.0, 1.0), wo.z());
                let dielectric_specular_component = (1.0 - metallic) * dielectric_fresnel_mix * Vec3(1.0, 1.0, 1.0);
                return BsdfSample { 
                    wi: Vec3(-wo.0, -wo.1, wo.2), 
                    bsdf: metal_component + dielectric_specular_component, 
                    pdf: specular_prob, // includes delta
                    specular: true,
                }
            }
            else {
                let (wi, wi_pdf) = sample::sample_cosine_hemisphere();
                return BsdfSample { 
                    wi, 
                    bsdf: (1.0 - specular_prob) * base_color * f32::consts::FRAC_1_PI, 
                    pdf: (1.0 - specular_prob) * wi_pdf, 
                    specular: false, 
                }
            }
        }

        let wm = super::microfacet::sample_wm(wo, alpha, alpha);
        let specular_wi = Vec3::reflect(wo, wm);
        let (diffuse_wi, _) = sample::sample_cosine_hemisphere();

        let dielectric_fresnel_mix = fresnel_mix_factor(1.5, Vec3::dot(wo, wm));

        // we stochastically sample the direction according to mix probabilities
        // however, our pdf needs to account for all the ways that a direction could've been chosen
        let wi = if sample::sample_uniform() < metallic {
            // metal
            specular_wi
        }
        else {
            // dielectric
            if sample::sample_uniform() < dielectric_fresnel_mix {
                // reflection
                specular_wi
            }
            else {
                // diffuse
                diffuse_wi
            }
        };

        // invalid sample (higher-order scattering event)
        // don't waste effort computing actual bsdf / pdf
        if wi.z() < 0.0 {
            return BsdfSample { wi, bsdf: Vec3::zero(), pdf: 1.0, specular: false, }
        }

        let bsdf = bsdf(
            wo,
            wi,
            base_color,
            metallic,
            alpha
        );

        let pdf = pdf(
            wo,
            wi,
            metallic,
            alpha
        );

        BsdfSample { wi, bsdf, pdf, specular: false }
    }
}