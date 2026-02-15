use std::{f32, hash::{Hash, Hasher}};

use bitflags::bitflags;
use raytracing::{geometry::{Complex, Vec2, Vec3}, materials::Material, scene};
use rustc_hash::FxHasher;
use tracing::warn;

use crate::{accel, materials::microfacet::{torrance_sparrow_pdf, torrance_sparrow_refl_pdf}, ray::{Ray, RayDifferentials}, sample::{self, CpuSampler, power_heuristic, sample_exponential}, texture::CpuTextures};

#[derive(Debug)]
pub(crate) struct BsdfSample {
    pub(crate) wi: Vec3,
    pub(crate) bsdf: Vec3,
    pub(crate) pdf: f32,

    // which component this sample represents
    // only one flag should be set; checked by conversion to ValidBsdfSample
    pub(crate) component: BsdfComponentFlags,
}

// helper to prevent bad values from propagating into render output from BSDF evaluation
pub(crate) enum ValidBsdfSample {
    ValidSample(BsdfSample),
    ValidNullSample,
    InvalidSample
}

impl From<BsdfSample> for ValidBsdfSample {
    fn from(value: BsdfSample) -> Self {
        let bad = |f: f32| f.is_infinite() || f.is_nan();
        let bad_vec = |v: Vec3| bad(v.0) || bad(v.1) || bad(v.2);
        if bad_vec(value.bsdf) || bad(value.pdf) || value.pdf <= 0.0 || bad_vec(value.wi) || value.component.bits().count_ones() != 1 {
            warn!("bad bsdfsample generated");
            return Self::InvalidSample;
        }

        Self::ValidSample(value)
    }
}

// Corresponds to `Material` evaluated at a specific point
#[derive(Debug)]
pub(crate) enum CpuBsdf {
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

    LayeredBsdf(Box<LayeredBsdf>)
}

#[derive(Debug)]
pub(crate) struct LayeredBsdf {
    top: CpuBsdf,
    bottom: CpuBsdf,
    n_samples: u32,
    max_depth: u32,
    thickness: f32,
    albedo: Vec3,
    g: f32,
}

impl LayeredBsdf {
    #[allow(non_snake_case, reason = "physics convention")]
    fn Tr(dz: f32, w: Vec3) -> f32 {
        let distance = f32::abs(dz / w.z());
        f32::exp(-distance)
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct BsdfComponentFlags : u8 {
        const NONSPECULAR_REFLECTION = 1 << 0;
        const SPECULAR_REFLECTION = 1 << 1;
        const NONSPECULAR_TRANSMISSION = 1 << 2;
        const SPECULAR_TRANSMISSION = 1 << 3;
        
        const REFLECTION = BsdfComponentFlags::NONSPECULAR_REFLECTION.bits() | BsdfComponentFlags::SPECULAR_REFLECTION.bits();
        const TRANSMISSION = BsdfComponentFlags::NONSPECULAR_TRANSMISSION.bits() | BsdfComponentFlags::SPECULAR_TRANSMISSION.bits();
        const SPECULAR = BsdfComponentFlags::SPECULAR_REFLECTION.bits() | BsdfComponentFlags::SPECULAR_TRANSMISSION.bits();
        const NONSPECULAR = BsdfComponentFlags::NONSPECULAR_REFLECTION.bits() | BsdfComponentFlags::NONSPECULAR_TRANSMISSION.bits();
    }
}

impl BsdfComponentFlags {
    pub(crate) fn is_specular(self) -> bool {
        self.intersects(Self::SPECULAR)
    }

    pub(crate) fn is_nonspecular(self) -> bool {
        self.intersects(Self::NONSPECULAR)
    }

    pub(crate) fn is_reflection(self) -> bool {
        self.intersects(Self::REFLECTION)
    }

    pub(crate) fn is_transmission(self) -> bool {
        self.intersects(Self::TRANSMISSION)
    }
}

impl CpuBsdf {
    // local coordinates, shading normal along +z
    pub(crate) fn evaluate_bsdf(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        match self {
            CpuBsdf::Diffuse { albedo } => {
                if wo.z() * wi.z() < 0.0 {
                    Vec3::zero()
                } else {
                    *albedo / f32::consts::PI
                }
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

            CpuBsdf::LayeredBsdf(inner) => {
                let LayeredBsdf { 
                    top, 
                    bottom,
                    n_samples, 
                    max_depth,
                    thickness, 
                    albedo,
                    g 
                } = inner.as_ref();
                let (n_samples, max_depth, thickness, albedo, g) = (*n_samples, *max_depth, *thickness, *albedo, *g);

                let mut f = Vec3::zero();
                
                // we will always assume material to be two-sided (i.e. incident ray always impinges on top layer, regardless of surface normal)
                let (wo, wi) = if wo.z() < 0.0 {
                    (-wo, -wi)
                }
                else {
                    (wo, wi)
                };

                let enter_interface = top;
                let (exit_interface, non_exit_interface, exit_z) = if wi.z() < 0.0 {
                    (bottom, top, 0.0)
                } else {
                    (top, bottom, thickness)
                };

                if !enter_interface.components().is_transmission() || !exit_interface.components().is_transmission() {
                    return Vec3::zero();
                }

                // account for singular reflection
                if wo.z() * wi.z() > 0.0 {
                    f += (n_samples as f32) * enter_interface.evaluate_bsdf(wo, wi);
                }

                let mut hasher = FxHasher::default();
                wi.hash(&mut hasher);
                wo.hash(&mut hasher);
                
                let seed = hasher.finish();
                let mut sampler = CpuSampler::one_off_sampler(seed);

                for _ in 0..n_samples {
                    // TODO: this needs to account for non-symmetry
                    // dbg!(&enter_interface);
                    // dbg!(&exit_interface);
                    let ValidBsdfSample::ValidSample(enter) = enter_interface.sample_bsdf(wo, BsdfComponentFlags::TRANSMISSION, &mut sampler) else {
                        continue;
                    };

                    let ValidBsdfSample::ValidSample(exit) = exit_interface.sample_bsdf(wi, BsdfComponentFlags::TRANSMISSION, &mut sampler) else {
                        continue;
                    };

                    let mut beta = exit.bsdf * f32::abs(exit.wi.z()) / exit.pdf;
                    let mut z = thickness;
                    let mut w = enter.wi;

                    for depth in 0..max_depth {
                        if depth > 3 && beta.max_component() < 0.25 {
                            let q = f32::max(0.0, beta.max_component());
                            if sampler.sample_uniform() < q {
                                break;
                            }

                            beta /= 1.0 - q
                        }

                        if albedo == Vec3::zero() {
                            z = if z == thickness { 0.0 } else { thickness };
                            beta *= LayeredBsdf::Tr(thickness, w);
                        }
                        else {
                            let sigma_t = 1.0;
                            let dz = sample_exponential(sampler.sample_uniform(), sigma_t / f32::abs(w.z()));
                            let zp = if w.z() > 0.0 { z + dz } else { z - dz };

                            if 0.0 < zp && zp < thickness {
                                // we sampled a scattering event between the two layers
                                // we use MIS between the distribution of directions from exit interface's BTDF and that given by phase function
                                let wt = if exit_interface.is_delta_bsdf() {
                                    1.0
                                } else {
                                    sample::power_heuristic(1, exit.pdf, 1, phase_function::pdf(-w, -exit.wi, g))
                                };

                                f += beta * albedo * phase_function::p(-w, -exit.wi, g) * wt * LayeredBsdf::Tr(zp - exit_z, exit.wi) * exit.bsdf / exit.pdf;
                                
                                let u = sampler.sample_uniform2();
                                let phase_sample = phase_function::sample_p(-w, g, u);
                                
                                beta *= albedo * phase_sample.p / phase_sample.pdf;
                                w = phase_sample.wi;
                                z = zp;

                                let facing_exit = (z < exit_z && w.z() > 0.0) || (z > exit_z && w.z() < 0.0);
                                if !exit_interface.is_delta_bsdf() && facing_exit {
                                    let exit_f = exit_interface.evaluate_bsdf(-w, wi);
                                    if exit_f != Vec3::zero() {
                                        let exit_pdf = exit_interface.evaluate_pdf(-w, wi, BsdfComponentFlags::TRANSMISSION);
                                        let wt = sample::power_heuristic(1, phase_sample.pdf, 1, exit_pdf);

                                        f += beta * LayeredBsdf::Tr(zp - exit_z, phase_sample.wi) * exit_f * wt;
                                    }
                                }

                                continue;
                            }
                            z = f32::clamp(zp, 0.0, thickness);
                        }

                        if z == exit_z {
                            let ValidBsdfSample::ValidSample(exit_reflect_sample) = exit_interface.sample_bsdf(-w, BsdfComponentFlags::REFLECTION, &mut sampler) else {
                                break;
                            };

                            beta *= exit_reflect_sample.bsdf * exit_reflect_sample.wi.z().abs() / exit_reflect_sample.pdf;
                            w = exit_reflect_sample.wi;
                        }
                        else {
                            if !non_exit_interface.is_delta_bsdf() {
                                let wt = if non_exit_interface.is_delta_bsdf() {
                                    1.0
                                }
                                else {
                                    power_heuristic(1, exit.pdf, 1, non_exit_interface.evaluate_pdf(-w, -exit.wi, BsdfComponentFlags::REFLECTION))
                                };

                                f += beta * non_exit_interface.evaluate_bsdf(-w, -exit.wi) * exit.wi.z().abs() * wt * LayeredBsdf::Tr(thickness, exit.wi) * exit.bsdf / exit.pdf;
                            }

                            // sample new direction from non_exit_interface
                            let ValidBsdfSample::ValidSample(non_exit_sample) = non_exit_interface.sample_bsdf(-w, BsdfComponentFlags::REFLECTION, &mut sampler) else {
                                break
                            };

                            beta *= non_exit_sample.bsdf * non_exit_sample.wi.z().abs() / non_exit_sample.pdf;
                            w = non_exit_sample.wi;

                            if !exit_interface.is_delta_bsdf() {
                                let exit_f  = exit_interface.evaluate_bsdf(-w, wi);
                                if exit_f != Vec3::zero() {
                                    let exit_pdf = exit_interface.evaluate_pdf(-w, wi, BsdfComponentFlags::all());
                                    let wt = if non_exit_interface.is_delta_bsdf() {
                                        1.0
                                    }
                                    else {
                                        power_heuristic(1, non_exit_sample.pdf, 1, exit_pdf)
                                    };

                                    f += beta * LayeredBsdf::Tr(thickness, non_exit_sample.wi) * exit_f * wt;
                                }
                            }
                            
                        }
                    }
                }

                f / (n_samples as f32)
            }
            
        }
    }

    pub(crate) fn evaluate_pdf(&self, wo: Vec3, wi: Vec3, component: BsdfComponentFlags) -> f32 {
        match self {
            CpuBsdf::Diffuse { .. } => {
                if !component.contains(BsdfComponentFlags::NONSPECULAR_REFLECTION) {
                    return 0.0;
                }

                if wo.z() * wi.z() > 0.0 {
                    1.0 / (2.0 * f32::consts::PI)
                }
                else {
                    0.0
                }
            },

            // see note in `evaluate_bsdf`
            CpuBsdf::SmoothDielectric { .. } => 0.0,
            CpuBsdf::SmoothConductor { .. } => 0.0,

            CpuBsdf::RoughConductor { alpha_x, alpha_y, .. } => {
                if !component.contains(BsdfComponentFlags::NONSPECULAR_REFLECTION) {
                    return 0.0;
                }

                torrance_sparrow_refl_pdf(wo, wi, *alpha_x, *alpha_y)
            },
            CpuBsdf::RoughDielectric { eta, alpha_x, alpha_y } => {    
                torrance_sparrow_pdf(wo, wi, *eta, *alpha_x, *alpha_y, component)
            },

            CpuBsdf::LayeredBsdf(_) => todo!("implement pdf for LayeredBsdf"),
        }
    }

    // we aim to consistently sample the same number of dimensions when the underlying material
    // is the same to maximize the benefits of stratification. however, some materials simply require
    // more samples than others, so only *within* a single material (which will likely lead to similar 
    // subsequent bsdf sampling / similar light sampling later) do we try to ensure this; otherwise
    // every material would take the same number of dimensions as the maximum of all materials
    pub(crate) fn sample_bsdf(
        &self, 
        wo: Vec3,
        component: BsdfComponentFlags,
        sampler: &mut CpuSampler
    ) -> ValidBsdfSample {
        match self {
            CpuBsdf::Diffuse { albedo } => {
                if !component.contains(BsdfComponentFlags::NONSPECULAR_REFLECTION) {
                    // warn!("invalid bsdf component for diffuse bsdf");
                    // dbg!(component);
                    // panic!();
                    return ValidBsdfSample::InvalidSample;
                }

                // cosine-weighted hemisphere sampling
                let u = sampler.sample_uniform2();
                let wi = sample::sample_cosine_hemisphere(u);
                let pdf = wi.z() / f32::consts::PI;
                
                BsdfSample {
                    wi,
                    bsdf: *albedo / f32::consts::PI,
                    pdf,
                    component: BsdfComponentFlags::NONSPECULAR_REFLECTION,
                }.into()
            },

            CpuBsdf::SmoothDielectric { eta } => {
                if !component.intersects(BsdfComponentFlags::SPECULAR_REFLECTION | BsdfComponentFlags::SPECULAR_TRANSMISSION) {
                    warn!("invalid bsdf component for smooth dielectric");
                    return ValidBsdfSample::InvalidSample;
                }

                let normal = Vec3(0.0, 0.0, 1.0);

                #[allow(non_snake_case, reason = "physics convention")]
                let R = fresnel_dielectric(wo.z(), *eta);
                
                #[allow(non_snake_case, reason = "physics convention")]
                let T = 1.0 - R;

                let p_reflect = if component.contains(BsdfComponentFlags::SPECULAR_REFLECTION) { R } else { 0.0 };
                let p_transmit = if component.contains(BsdfComponentFlags::SPECULAR_TRANSMISSION) { T } else { 0.0 };

                let p_total = p_reflect + p_transmit;

                // we randomly choose to sample the Dirac in the reflected direction,
                // or in the refracted direction, proportional to calculated reflection
                // and transmission coefficients
                let sample = sampler.sample_uniform();
                let bsdf_sample = if sample * p_total < p_reflect {
                    let reflection_dir = Vec3::reflect(wo, normal);
                    let cos_theta = reflection_dir.z().abs();
                    let f = R / cos_theta;
                    let pdf = R / p_total;
                    
                    BsdfSample {
                        wi: reflection_dir,
                        bsdf: Vec3(f, f, f),
                        pdf,
                        component: BsdfComponentFlags::SPECULAR_REFLECTION,
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
                        
                        return ValidBsdfSample::InvalidSample;
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

                    let pdf = T / p_total;
                    BsdfSample {
                        wi: refract_dir,
                        bsdf: Vec3(f, f, f),
                        pdf,
                        component: BsdfComponentFlags::SPECULAR_TRANSMISSION,
                    }
                };

                bsdf_sample.into()
            },

            CpuBsdf::SmoothConductor { eta, kappa } => {
                if !component.contains(BsdfComponentFlags::SPECULAR_REFLECTION) {
                    warn!("invalid bsdf component for smooth conductor");
                }

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
                    component: BsdfComponentFlags::SPECULAR_REFLECTION
                }.into()
            },

            CpuBsdf::RoughConductor { 
                eta, 
                kappa,
                alpha_x, 
                alpha_y 
            } => {
                if !component.contains(BsdfComponentFlags::REFLECTION) {
                    warn!("invalid bsdf component for rough conductor");
                }

                microfacet::torrance_sparrow_refl_sample(
                    wo, 
                    *eta, 
                    *kappa,
                    *alpha_x, 
                    *alpha_y,
                    sampler,
                )
            },

            CpuBsdf::RoughDielectric { 
                eta, 
                alpha_x, 
                alpha_y 
            } => {
                if component.is_empty() {
                    warn!("invalid bsdf component for rough dielectric")
                }

                microfacet::torrance_sparrow_sample(
                    wo, 
                    *eta, 
                    *alpha_x, 
                    *alpha_y,
                    component,
                    sampler,
                )
            },

            CpuBsdf::LayeredBsdf(inner) => {
                let LayeredBsdf { 
                    top, 
                    bottom, 
                    n_samples: _, 
                    max_depth, 
                    thickness, 
                    albedo, 
                    g 
                } = inner.as_ref();
                let (&max_depth, &thickness, &albedo, &g) = (max_depth, thickness, albedo, g);
                let (wo, flip_wi) = if wo.z() < 0.0 {
                    (-wo, true)
                } else {
                    (wo, false)
                };

                let enter_sample = top.sample_bsdf(wo, BsdfComponentFlags::all(), sampler);
                let ValidBsdfSample::ValidSample(enter_sample) = enter_sample else {
                    return enter_sample;
                };

                if enter_sample.component.is_reflection() {
                    return BsdfSample {
                        wi: if flip_wi { -enter_sample.wi } else { enter_sample.wi },
                        ..enter_sample
                    }.into();
                }

                let mut specular_path = enter_sample.component.is_specular();

                let mut w = enter_sample.wi;
                let mut f = enter_sample.bsdf * enter_sample.wi.z().abs();
                let mut pdf = enter_sample.pdf;
                let mut z = thickness;

                for depth in 0..max_depth {
                    let rr_beta = f.max_component() / pdf;
                    if depth > 3 && rr_beta < 0.25 {
                        let q = f32::max(0.0, 1.0 - rr_beta);
                        if sampler.sample_uniform() < q {
                            return ValidBsdfSample::ValidNullSample;
                        }

                        pdf *= 1.0 - q;
                    }

                    if w.z() == 0.0 {
                        return ValidBsdfSample::ValidNullSample;
                    }
        
                    if albedo != Vec3::zero() {
                        // Sample potential scattering event in layered medium
                        let sigma_t = 1.0;
                        let dz = sample::sample_exponential(sampler.sample_uniform(), sigma_t / w.z().abs());
                        let zp = if w.z() > 0.0 { z + dz } else { z - dz };
                        
                        if zp == z {
                            warn!("sampled zero distance in layeredbsdf");
                            return ValidBsdfSample::InvalidSample;
                        }

                        if 0.0 < zp && zp < thickness {
                            // Update path state for valid scattering event between interfaces
                            let phase_sample = phase_function::sample_p(-w, g, sampler.sample_uniform2());
                            if phase_sample.wi.z() == 0.0 {
                                return ValidBsdfSample::ValidNullSample;
                            }
                            
                            f *= albedo * phase_sample.p;
                            pdf *= phase_sample.pdf;
                            specular_path = false;
                            w = phase_sample.wi;
                            assert!(f32::abs(w.length() - 1.0) < 1.0e-4);
                            z = zp;
        
                            continue;
                        }
                        z = f32::clamp(zp, 0.0, thickness);
                        
                    } else {
                        z = if z == thickness { 0.0 } else { thickness };
                        f *= LayeredBsdf::Tr(thickness, w);
                    }
                    
                    let interface = if z == 0.0 { bottom } else { top };
        
                    // Sample interface BSDF to determine new path direction
                    assert!(f32::abs(w.length() - 1.0) < 1.0e-4);
                    let interface_sample = interface.sample_bsdf(-w, BsdfComponentFlags::all(), sampler);
                    let ValidBsdfSample::ValidSample(interface_sample) = interface_sample else {
                        return interface_sample;
                    };

                    f *= interface_sample.bsdf;
                    pdf *= interface_sample.pdf;
                    specular_path &= interface_sample.component.is_specular();
                    w = interface_sample.wi;
        
                    // Return sample if path has left the layers
                    if interface_sample.component.is_transmission() {
                        let same_direction = wo.z() * w.z() > 0.0;
                        let sampled_component = match (same_direction, specular_path) {
                            (true, true) => BsdfComponentFlags::SPECULAR_REFLECTION,
                            (true, false) => BsdfComponentFlags::NONSPECULAR_REFLECTION,
                            (false, true) => BsdfComponentFlags::SPECULAR_TRANSMISSION,
                            (false, false) => BsdfComponentFlags::NONSPECULAR_TRANSMISSION,
                        };

                        if flip_wi {
                            w = -w;
                        }
                        return BsdfSample {
                            wi: w,
                            bsdf: f,
                            pdf,
                            component: sampled_component,
                        }.into();
                    }
        
                    // Scale f by cosine term after scattering at the interface
                    f *= interface_sample.wi.z().abs();
                }

                // didn't escape layer sandwich before hitting max_depth
                ValidBsdfSample::ValidNullSample
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
            CpuBsdf::LayeredBsdf { .. } => false,
        }
    }

    pub(crate) fn components(&self) -> BsdfComponentFlags {
        match self {
            CpuBsdf::Diffuse { .. } => BsdfComponentFlags::NONSPECULAR_REFLECTION,
            CpuBsdf::SmoothDielectric { .. } => BsdfComponentFlags::SPECULAR_REFLECTION | BsdfComponentFlags::SPECULAR_TRANSMISSION,
            CpuBsdf::SmoothConductor { .. } => BsdfComponentFlags::SPECULAR_REFLECTION,
            CpuBsdf::RoughConductor { .. } => BsdfComponentFlags::NONSPECULAR_REFLECTION,
            CpuBsdf::RoughDielectric { .. } => BsdfComponentFlags::NONSPECULAR_REFLECTION | BsdfComponentFlags::NONSPECULAR_TRANSMISSION,
            CpuBsdf::LayeredBsdf(_) => todo!(),
        }
    }
}

// we use dpdu / dpdv from parametric intersection tests with either:
// 1. camera ray differentials for primary rays
// 2. fake camera ray differentials for non-primary rays
// in order to compute estimates on the texture-space footprint of a single ray
// so that the appropriate mip-level can be sampled for antialiasing purposes
// references:
// - pbrt 4ed 10.1
// - YKL's "Mipmapping with Bidirectional Techniques" article
#[derive(Debug, Clone)]
pub(crate) struct MaterialEvalContext {
    pub(crate) uv: Vec2,

    pub(crate) dudx: f32,
    pub(crate) dudy: f32,
    pub(crate) dvdx: f32,
    pub(crate) dvdy: f32,
}

impl MaterialEvalContext {
    // computes d{u,v}/d{x,y} from dp/d{x,y} and dp/d{u,v}
    // using the chain rule and least-squares
    fn new(
        hit_info: &accel::HitInfo,
        dpdx: Vec3,
        dpdy: Vec3,
    ) -> Self {
        let dpdu = hit_info.dpdu;
        let dpdv = hit_info.dpdv;
        
        let ata00 = Vec3::dot(dpdu, dpdu);
        let ata11 = Vec3::dot(dpdv, dpdv);
        let ata01 = Vec3::dot(dpdu, dpdv);

        let det = ata00 * ata11 - ata01 * ata01;
        let inv_det = 1.0 / det;
        
        let atb0x = Vec3::dot(dpdu, dpdx);
        let atb1x = Vec3::dot(dpdv, dpdx);
        let atb0y = Vec3::dot(dpdu, dpdy);
        let atb1y = Vec3::dot(dpdv, dpdy);

        let dudx = inv_det * (ata11 * atb0x - ata01 * atb1x);
        let dvdx = inv_det * (ata00 * atb1x - ata01 * atb0x);
        let dudy = inv_det * (ata11 * atb0y - ata01 * atb1y);
        let dvdy = inv_det * (ata00 * atb1y - ata01 * atb0y);

        // clamp to reasonable values: if d{u,v}/d{x,y} is 0,
        // it should just revert to point sampling which is always "safe"
        let clamp = |v: f32| {
            if f32::is_finite(v) { 
                f32::clamp(v, -1.0e8, 1.0e8) 
            } 
            else { 
                0.0 
            }
        };

        let dudx = clamp(dudx);
        let dudy = clamp(dudy);
        let dvdx = clamp(dvdx);
        let dvdy = clamp(dvdy);

        MaterialEvalContext { 
            uv: hit_info.uv, 
            dudx, 
            dudy, 
            dvdx, 
            dvdy 
        }
    }

    pub(crate) fn new_without_antialiasing(
        uv: Vec2
    ) -> Self {
        MaterialEvalContext { uv, dudx: 0.0, dudy: 0.0, dvdx: 0.0, dvdy: 0.0 }
    }

    pub(crate) fn new_from_ray_differentials(
        hit_info: &accel::HitInfo, // world-space
        ray: Ray,
        ray_differentials: RayDifferentials, // world-space
    ) -> Self {
        // hit_info.point + hit_info.normal defines a plane, which we intersect
        // x and y ray differentials against
        let n = hit_info.normal;
        let p = hit_info.point;
        let rx_o = ray.origin + ray_differentials.x_origin;
        let rx_d = ray.direction + ray_differentials.x_direction;
        let ry_o = ray.origin + ray_differentials.y_origin;
        let ry_d = ray.direction + ray_differentials.y_direction;

        let d = -Vec3::dot(n, p);
        let tx = -(Vec3::dot(n, rx_o) + d) / Vec3::dot(n, rx_d);
        let ty = -(Vec3::dot(n, ry_o) + d) / Vec3::dot(n, ry_d);
        
        let px = rx_o + tx * rx_d;
        let py = ry_o + ty * ry_d;

        let dpdx = px - hit_info.point;
        let dpdy = py - hit_info.point;

        MaterialEvalContext::new(hit_info, dpdx, dpdy)
    }

    pub(crate) fn new_from_camera(
        hit_info: &accel::HitInfo,
        camera: &scene::Camera,
        min_ray_differentials: &RayDifferentials,
        spp: u32,
    ) -> Self {
        let dpdx = todo!();
        let dpdy = todo!();
        
        MaterialEvalContext::new(hit_info, dpdx, dpdy)
    }
}

pub(crate) trait CpuMaterial {
    // evaluate the material at a specific shading point to get a bsdf
    fn get_bsdf(&self, eval_ctx: &MaterialEvalContext, textures: &CpuTextures) -> CpuBsdf;

    // evaluate what mip-level the "primary" (usually albedo / color) texture
    // associated with a material is, or None if it's not applicable (i.e. it's not an image texture)
    fn get_mip_level(&self, eval_ctx: &MaterialEvalContext, textures: &CpuTextures) -> Option<f32>;

    // evaluate albedo (for denoising)
    fn get_albedo(&self, eval_ctx: &MaterialEvalContext, textures: &CpuTextures) -> Vec3;
}

impl CpuMaterial for Material {
    fn get_bsdf(&self, eval_ctx: &MaterialEvalContext, textures: &CpuTextures) -> CpuBsdf {
        match self {
            Material::Diffuse { albedo } => {
                let albedo = textures.sample(*albedo, eval_ctx);
                let albedo = Vec3(albedo.r(), albedo.g(), albedo.b());
                
                CpuBsdf::Diffuse { albedo }
            },
            Material::SmoothDielectric { eta } => {
                let eta = textures.sample(*eta, eval_ctx).r();

                CpuBsdf::SmoothDielectric { eta }
            },
            Material::SmoothConductor { eta, kappa } => {
                let eta = textures.sample(*eta, eval_ctx);
                let kappa = textures.sample(*kappa, eval_ctx);
                let eta = Vec3(eta.r(), eta.g(), eta.b());
                let kappa = Vec3(kappa.r(), kappa.g(), kappa.b());

                CpuBsdf::SmoothConductor { eta, kappa }
            },

            Material::RoughConductor { eta, kappa, roughness, remap_roughness } => {
                let eta = textures.sample(*eta, eval_ctx);
                let kappa = textures.sample(*kappa, eval_ctx);
                let eta = Vec3(eta.0, eta.1, eta.2);
                let kappa = Vec3(kappa.0, kappa.1, kappa.2);

                let roughness = textures.sample(*roughness, eval_ctx);
                let mut alpha_x = roughness.0;
                let mut alpha_y = roughness.1;

                if *remap_roughness {
                    alpha_x = alpha_x.sqrt();
                    alpha_y = alpha_y.sqrt();
                }

                if f32::max(alpha_x, alpha_y) < consts::MINIMUM_ROUGHNESS {
                    CpuBsdf::SmoothConductor { 
                        eta, 
                        kappa 
                    }
                }
                else {
                    CpuBsdf::RoughConductor { 
                        eta, 
                        kappa,
                        alpha_x, 
                        alpha_y 
                    }
                }
            }

            Material::RoughDielectric { eta, roughness, remap_roughness } => {
                let eta = textures.sample(*eta, eval_ctx).r();

                let roughness = textures.sample(*roughness, eval_ctx);
                let mut alpha_x = roughness.0;
                let mut alpha_y = roughness.1;

                if *remap_roughness {
                    alpha_x = alpha_x.sqrt();
                    alpha_y = alpha_y.sqrt();
                }

                if f32::max(alpha_x, alpha_y) < consts::MINIMUM_ROUGHNESS {
                    CpuBsdf::SmoothDielectric { eta }
                }
                else {
                    CpuBsdf::RoughDielectric { eta, alpha_x, alpha_y }
                }
            }

            Material::CoatedDiffuse { 
                diffuse_albedo, 
                dielectric_eta, 
                dielectric_remap_roughness,
                dielectric_roughness, 
                thickness, 
                coat_albedo 
            } => {
                let diffuse_albedo = {
                    let v = textures.sample(*diffuse_albedo, eval_ctx);
                    Vec3(v.0, v.1, v.2)
                };

                let bottom = CpuBsdf::Diffuse { albedo: diffuse_albedo };
                
                let dielectric_eta = textures.sample(*dielectric_eta, eval_ctx).0;
                let top = if let Some(dielectric_roughness) = *dielectric_roughness {
                    let roughness = textures.sample(dielectric_roughness, eval_ctx);

                    let mut alpha_x = roughness.0;
                    let mut alpha_y = roughness.1;
                    if *dielectric_remap_roughness {
                        alpha_x = alpha_x.sqrt();
                        alpha_y = alpha_y.sqrt();
                    }

                    if f32::max(alpha_x, alpha_y) < consts::MINIMUM_ROUGHNESS {
                        CpuBsdf::SmoothDielectric { eta: dielectric_eta }    
                    }
                    else {
                        CpuBsdf::RoughDielectric { eta: dielectric_eta, alpha_x, alpha_y }
                    }
                }
                else {
                    CpuBsdf::SmoothDielectric { eta: dielectric_eta }
                };

                let thickness = textures.sample(*thickness, eval_ctx).0;
                let coat_albedo = {
                    let v = textures.sample(*coat_albedo, eval_ctx);
                    Vec3(v.0, v.1, v.2)
                };

                let layered_bsdf = LayeredBsdf {
                    top,
                    bottom,
                    n_samples: 8,
                    max_depth: 8,
                    thickness,
                    albedo: coat_albedo,
                    g: 0.0,
                };

                CpuBsdf::LayeredBsdf(Box::new(layered_bsdf))
            }

            _ => todo!("support new material")
        }
    }

    fn get_mip_level(&self, eval_ctx: &MaterialEvalContext, textures: &CpuTextures) -> Option<f32> {
        let &main_tex = match self {
            Material::Diffuse { albedo } => albedo,
            Material::SmoothDielectric { .. } => return None,
            Material::SmoothConductor { .. } => return None,
            Material::RoughDielectric { .. } => return None,
            Material::RoughConductor { .. } => return None,
            Material::CoatedDiffuse { .. } => return None,
        };

        textures.texture_mip_level(main_tex, eval_ctx)
    }
    
    fn get_albedo(&self, eval_ctx: &MaterialEvalContext, textures: &CpuTextures) -> Vec3 {
        let albedo_tex = match self {
            Material::Diffuse { albedo } => Some(albedo),
            Material::SmoothDielectric { .. } => None,
            Material::SmoothConductor { .. } => None,
            Material::RoughDielectric { .. } => None,
            Material::RoughConductor { .. } => None,
            Material::CoatedDiffuse { diffuse_albedo, .. } => Some(diffuse_albedo), // TODO: would be good to take into account the transmittance of coat layer / albedo from top layer
        };

        if let Some(&albedo_tex) = albedo_tex {
            let albedo = textures.sample(albedo_tex, eval_ctx);
            Vec3(albedo.0, albedo.1, albedo.2)
        }
        else {
            Vec3(1.0, 1.0, 1.0)
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

#[test]
fn test_refract() {

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

    use crate::{materials::{BsdfComponentFlags, BsdfSample, ValidBsdfSample, fresnel_complex, fresnel_dielectric}, sample::{self, CpuSampler}};

    // PBRT 4ed 9.6.1
    // represents relative differential area of microfacets pointing in a particular direction
    // when microfacets distributed according to "ellipsoidal bumps"
    // as usual, wm is in shading coordinate frame (+z <=> normal of "macrosurface")
    fn distribution(wm: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
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
    fn lambda(w: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
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
    fn G1(w: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
        1.0 / (1.0 + lambda(w, alpha_x, alpha_y))
    }

    // PBRT 4ed 9.6.3
    // an approximation to the masking-shadowing function (shadowing is masking but in the outgoing direction)
    #[allow(non_snake_case, reason = "physics convention")]
    fn G(wo: Vec3, wi: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
        1.0 / (1.0 + lambda(wo, alpha_x, alpha_y) + lambda(wi, alpha_x, alpha_y))
    }

    // PBRT 4ed 9.6.4
    // distribution of normals which are visible from some direction
    // (leads to improved sampling efficiency)
    fn visible_distribution(w: Vec3, wm: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
        let cos_theta = w.z().abs();

        (G1(w, alpha_x, alpha_y) / cos_theta) * 
        distribution(wm, alpha_x, alpha_y) * 
        Vec3::dot(w, wm).abs()
    }

    // PBRT 4ed 9.6.4
    // sample from visible normals distribution
    // figure 9.29 visually represents what this is doing
    fn sample_wm(w: Vec3, alpha_x: f32, alpha_y: f32, u: Vec2) -> Vec3 {
        let mut wh = Vec3(alpha_x * w.x(), alpha_y * w.y(), w.z()).unit();
        if wh.z() < 0.0 { wh = -wh; }

        let p = sample::sample_unit_disk(u);
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

    fn check_roughness(alpha_x: f32, alpha_y: f32) {
        debug_assert!(alpha_x >= super::consts::MINIMUM_ROUGHNESS && alpha_y >= super::consts::MINIMUM_ROUGHNESS, "surface is too smooth; should be handled by smooth bsdfs");
    }

    // pdf
    pub(super) fn torrance_sparrow_refl_pdf(wo: Vec3, wi: Vec3, alpha_x: f32, alpha_y: f32) -> f32 {
        check_roughness(alpha_x, alpha_y);

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
        check_roughness(alpha_x, alpha_y);

        // protect against grazing angle reflections
        if (wo + wi) == Vec3::zero() { return Vec3::zero(); }
        let wm = (wo + wi).unit();
        let cos_theta = Vec3::dot(wm, wi);
        let fresnel = Vec3(
            fresnel_complex(cos_theta.abs(), Complex(eta.0, kappa.0)),
            fresnel_complex(cos_theta.abs(), Complex(eta.1, kappa.1)),
            fresnel_complex(cos_theta.abs(), Complex(eta.2, kappa.2))
        );

        distribution(wm, alpha_x, alpha_y)
        * fresnel
        * G(wo, wi, alpha_x, alpha_y)
        / (4.0 * wo.z() * wi.z())
    }

    pub(super) fn torrance_sparrow_refl_sample(
        wo: Vec3,
        eta: Vec3,
        kappa: Vec3,
        alpha_x: f32,
        alpha_y: f32,
        sampler: &mut CpuSampler,
    ) -> ValidBsdfSample {
        check_roughness(alpha_x, alpha_y);

        let u = sampler.sample_uniform2();
        let wm = sample_wm(wo, alpha_x, alpha_y, u);
        let wi = Vec3::reflect(wo, wm);
        
        // this represents the case that the ray bounces deeper into
        // the microsurface; we discard these samples because we only
        // account for first-order scattering events in microfacet model
        // pbrt notes that this leads to energy loss and is thus not fully physical
        if wo.z() * wi.z() < 0.0 {
            return ValidBsdfSample::ValidNullSample;
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
            component: BsdfComponentFlags::NONSPECULAR_REFLECTION
        }.into()
    }

    // PBRT 4ed 9.7
    pub(super) fn torrance_sparrow_pdf(
        wo: Vec3,
        wi: Vec3,
        eta: f32,
        alpha_x: f32,
        alpha_y: f32,
        component: BsdfComponentFlags,
    ) -> f32 {
        check_roughness(alpha_x, alpha_y);

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

        let p_reflect = if component.contains(BsdfComponentFlags::NONSPECULAR_REFLECTION) { R } else { 0.0 };
        let p_transmit = if component.contains(BsdfComponentFlags::NONSPECULAR_TRANSMISSION) { T } else { 0.0 };
        let p_total = p_reflect + p_transmit;

        if reflect {
            (p_reflect / p_total) * visible_distribution(wo, wm, alpha_x, alpha_y) / (4.0 * Vec3::dot(wo, wm).abs())
        }
        else {
            let denom = (Vec3::dot(wi, wm) + Vec3::dot(wo, wm) / eta_wm).powi(2);
            let dwm_dwi = Vec3::dot(wi, wm).abs() / denom;
            (p_transmit / p_total) * visible_distribution(wo, wm, alpha_x, alpha_y) * dwm_dwi
        }
    }

    pub(super) fn torrance_sparrow_bsdf(
        wo: Vec3,
        wi: Vec3,
        eta: f32,
        alpha_x: f32,
        alpha_y: f32
    ) -> Vec3 {
        check_roughness(alpha_x, alpha_y);

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
        component: BsdfComponentFlags,
        sampler: &mut CpuSampler,
    ) -> ValidBsdfSample {
        check_roughness(alpha_x, alpha_y);

        let u = sampler.sample_uniform2();
        let wm = sample_wm(wo, alpha_x, alpha_y, u);
         
        #[allow(non_snake_case, reason = "physics convention")]
        let R = fresnel_dielectric(Vec3::dot(wo, wm), eta);
        #[allow(non_snake_case, reason = "physics convention")]
        let T = 1.0 - R;

        let p_reflect = if component.contains(BsdfComponentFlags::REFLECTION) { R } else { 0.0 };
        let p_transmit = if component.contains(BsdfComponentFlags::TRANSMISSION) { T } else { 0.0 };

        let p_total = p_reflect + p_transmit;

        let (wi, reflected) = if sampler.sample_uniform() * p_total < p_reflect {
            let wi = Vec3::reflect(wo, wm);

            // see analogous comment in `torrance_sparrow_refl_sample`
            if wo.z() * wi.z() < 0.0 {
                return ValidBsdfSample::ValidNullSample;
            }

            (wi, true)
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

                return ValidBsdfSample::InvalidSample;
            };

            if wo.z() * wi.z() > 0.0 || wi.z() == 0.0 {
                // this is analogous to check in reflection case, 
                // and leads to loss of energy
                return ValidBsdfSample::ValidNullSample;
            }

            (wi, false)
        };

        let pdf = torrance_sparrow_pdf(
            wo, 
            wi, 
            eta, 
            alpha_x, 
            alpha_y,
            component
        );

        let bsdf = torrance_sparrow_bsdf(
            wo,
            wi,
            eta,
            alpha_x,
            alpha_y
        );

        let chosen_component = if reflected {
            BsdfComponentFlags::NONSPECULAR_REFLECTION
        } else {
            BsdfComponentFlags::NONSPECULAR_TRANSMISSION
        };

        BsdfSample {
            wi,
            bsdf,
            pdf,
            component: chosen_component,
        }.into()
    }
}


mod phase_function {
    //! Implements the Henyey-Greenstein phase function
    //! (pbrt 4ed 11.3.1)
    
    use std::f32;

    use raytracing::geometry::{Vec2, Vec3};

    use crate::geometry;

    fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
        let denom = 1.0 + g * g + 2.0 * g * cos_theta;
        f32::consts::FRAC_1_PI * 0.25 * (1.0 - g * g) / (denom * f32::sqrt(denom))
    }

    #[derive(Debug, Clone, Copy)]
    pub(super) struct PhaseFunctionSample {
        pub(super) wi: Vec3,
        pub(super) p: f32,
        pub(super) pdf: f32
    }

    fn sample_henyey_greenstein(wo: Vec3, g: f32, u: Vec2) -> PhaseFunctionSample {
        let cos_theta = if f32::abs(g) < 1.0e-3 {
            1.0 - 2.0 * u.0
        }
        else {
            let term = (1.0 - g * g) / (1.0 + g - 2.0 * g * u.0);
            -1.0 / (2.0 * g) * (1.0 + g * g - term * term)
        };

        let phi = 2.0 * f32::consts::PI * u.1;
        let sin_theta = f32::sqrt(1.0 - cos_theta * cos_theta);

        let direction_wo = Vec3(f32::cos(phi) * sin_theta, f32::sin(phi) * sin_theta, cos_theta);
        let (wo_x, wo_y) = geometry::make_orthonormal_basis(wo);
        let wi = direction_wo.0 * wo_x + direction_wo.1 * wo_y + direction_wo.2 * wo;
        
        let p = henyey_greenstein(cos_theta, g);

        PhaseFunctionSample { 
            wi, 
            p,
            pdf: p // sampling distribution matches real distribution exactly
        }
    }

    pub(super) fn p(wo: Vec3, wi: Vec3, g: f32) -> f32 {
        henyey_greenstein(Vec3::dot(wo, wi), g)
    }

    pub(super) fn sample_p(wo: Vec3, g: f32, u: Vec2) -> PhaseFunctionSample {
        sample_henyey_greenstein(wo, g, u)
    }

    pub(super) fn pdf(wo: Vec3, wi: Vec3, g: f32) -> f32 {
        p(wo, wi, g)
    }
}

// Sometimes you just need some magic numbers
pub(crate) mod consts {
    // minimum value for alpha_x and alpha_y in T-R microfacet model before 
    // falling back to smooth BSDF evaluation
    pub(crate) const MINIMUM_ROUGHNESS: f32 = 1.0e-3;
}