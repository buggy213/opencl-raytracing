//! Primitives are the basic unit of rendering, a convention adopted from PBRT
//! There are three essential types of primitive, mirroring the design of PBRT
//! 1. BasicPrimitive
//!     - Contains material, area light, and shape
//! 2. TransformPrimitive
//!     - Contains index to different primitive, as well as a transformation matrix
//! 3. AggregatePrimitive
//!     - Contains a group of indices to other primitives
//!     - On all backends, AggregatePrimitive serves as a hint for where BVH should be constructed, 
//!     nesting AggregatePrimitive with other AggregatePrimitive implicitly defines a multi-level BVH
//!     - TODO: how to infer this from standard scene description formats? and should there be an option
//!     to have an aggregate which does not create an additional level in the BVH?
//! 
//! - Unlike PBRT, individual triangles are not considered shapes, and the smallest unit is an entire triangular mesh
//! at which a unique material is assigned is an entire triangular mesh
//! - For the common case where # triangles >> # materials, this design makes sense and is probably more efficient than
//! storing a pointer to Material for every single triangle; "logical" meshes with >1 material 
//! thus would need to be split up into multiple "physical" ones
//! - It is expected that most scene geometry will be (TransformPrimitive -> BasicPrimitive)
//! even when instancing is not used; one could save some computation (ray transformations) by pre-applying transform
//! to the underlying shape
//!     - If this is actually a useful optimization, could implement a pass on primitive list to optimize for this scenario
//! 
//! # Mapping onto CPU
//! - We construct a BVH for each aggregate primitive using Embree
//!     - Basic primitives are handled by just adding elements of the primitive (i.e. triangles) directly
//!     to the BVH
//!     - Transform primitives push / apply transform onto stack of transforms, then process referenced primitive
//!     - Aggregate primitives recursively construct BVH, which becomes a single leaf in outer BVH
//! - When constructing BVH with Embree, each "build primitive" has two associated values: geomID and primID
//!     - We associate geomID with index into aggregate
//!     - primID is the underlying index for shape included in BasicPrimitive that is the deepest descendant of 
//!     aggregate element.
//!     - primID is 0 if that descendant is another AggregatePrimitive

use crate::geometry::{Transform, Shape};
use crate::materials::Material;
use crate::lights::Light;

/// Index into the owning Scene's arrays
pub type MaterialIndex = u32;
pub type PrimitiveIndex = u32;
pub type AreaLightIndex = u32;
pub type TransformIndex = u32;

/// The main enum for all scene primitives
#[derive(Debug, Clone)]
pub enum Primitive {
    Basic(BasicPrimitive),
    Transform(TransformPrimitive),
    Aggregate(AggregatePrimitive),
}

/// A primitive with a shape, material, and optional area light
#[derive(Debug, Clone)]
pub struct BasicPrimitive {
    pub shape: Shape,
    pub material: MaterialIndex,
    pub area_light: Option<AreaLightIndex>,
}

/// A primitive that applies a transform to another primitive
#[derive(Debug, Clone)]
pub struct TransformPrimitive {
    pub primitive: PrimitiveIndex,
    pub transform: Transform,
}

/// A primitive that groups other primitives (e.g., for BVH)
#[derive(Debug, Clone)]
pub struct AggregatePrimitive {
    pub children: Vec<PrimitiveIndex>,
}