//! Primitives are the basic unit of rendering, a convention adopted from PBRT
//! There are three essential types of primitive
//! # BasicPrimitive
//! - Contains material, area light, and shape
//! # TransformPrimitive
//! - Contains index to different primitive, as well as a transformation matrix
//! # AggregatePrimitive
//! - Contains a group of indices to other primitives
//! 
//! # Mapping onto CPU
//! - AggregatePrimitive serves as a hint for where BVH should be constructed
//! - Unlike PBRT, individual triangles are not considered shapes, and the smallest unit is an entire triangular mesh
//! with one material
//! - For the common case where # triangles >> # materials, this design makes sense and is probably more efficient than
//! storing a pointer to Material for every single triangle; "logical" meshes thus would need to be split up into multiple "physical" ones
//! - Embree BVH leaves contain primID, geomID -> geomID maps to which primitive
//! - primID = triangle index for triangle mesh, 0 for all others
//! 