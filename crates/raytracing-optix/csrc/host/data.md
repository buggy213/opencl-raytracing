# Program Compilation
There is only two pipelines which are supported, which are hardcoded into the renderer. There really isn't too much point adding more flexibility here, since this isn't a game engine. 
Thus, the set of `optix-ir` files which are turned into modules, the program groups that the programs within these modules are collected into, and the pipeline options / module options / payload layout is all totally hardcoded. 
The two pipelines are the one used to generate AOVs, and the one used to actually produce beauty render. 
The primary artifact from this stage is the pipeline itself, and all of the program groups produced. I think it makes sense to bifurcate this into two types (`AovPipelineWrapper` and `PathtracerPipelineWrapper`) which contain all of the program groups associated with each pipeline. In addition, the functionality to populate the SBT will be different for the two pipelines, and probably have different data types / functions as well. The actual layout of each hitgroup record will be the same though (since AOV calculation still needs access to material properties / mesh geometry data). 

# Data Layout
Vertices and triangle indices live inside the GAS for triangle meshes. However, it is only possible to get the vertices of the triangle, and not the indices of those vertices, from OptiX (`optixGetTriangleVertexData`), so we also have to keep the index buffer around. Also, rather than storing the indices / shading normals / UV's in the same allocation as the GAS, it makes more sense for them to be kept in separate allocations, which are then referenced from the SBT. 

There is a 1-1-1 mapping between primitives and GAS's and SBT records for hitgroups, since the scene representation already requires that each primitive only have one material. We might want to relax this later...

SBT hitgroup records will have the following format

| Opaque Record Header |
|----------------------|
| struct MeshData      |
| struct Material      |

`struct MeshData` is defined as `struct MeshData { uint3* indices, float3* normals, float2* uvs }`. These are unused for non-triangle meshes, but the space could be useful for other procedural geometry possibly...
A mesh without normals / UVs has those respective pointers set to null. 

`struct Material` is defined to be a `union` over the different sets of `TextureId` required by different materials, which mirrors the definition used in scene description. 
The program already knows which material type it is (since there is one closest-hit program per material)

Overall, each record should be on the order of ~100 bytes at most, which should be very manageable. Keeping it from being too bloated is important, since shadow rays require a full SBT record, but have no need for `MeshData` or `MaterialData` (we can leave them totally uninitialized, even) and thus that space is wasted. This effectively leads to 2x memory consumption for the SBT, which is ok. 

## SBT indexing

From OptiX documentation:
```
sbt-index =
    sbt-instance-offset
    + (sbt-geometry-acceleration-structure-index * sbt-stride-from-trace-call)
    + sbt-offset-from-trace-call
```
Due to 1-1 relationship between primitive and GAS (i.e. no GAS contains more than 1 build input), then `sbt-geometry-acceleration-structure-index` is always 0, and `sbt-stride-from-trace-call` can be anything (though we set it as `RAY_TYPES_COUNT` to be safe). `sbt-offset-from-trace-call` is set to 0 or 1 depending on whether we are trying to trace a radiance-carrying ray or a shadow ray for next-event estimation of direct lighting. 

To actually index into the SBT, the hierarchy of IAS is constructed to point into it in a "flat" manner. Specifically, we don't make use of the fact that `sbt-instance-offset` accumulates as you traverse a hierarchy of IAS's. 

Since the AS-hierarchy has a 1-1 relationship with Primitive hierarchy in scene description, we can construct the two at the same time. 

# Textures


