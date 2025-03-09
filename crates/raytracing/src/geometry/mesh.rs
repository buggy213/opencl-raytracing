use std::{path::Path, ptr::null_mut, slice::from_raw_parts};

use obj::{ObjError, MtlLibsLoadError, Obj, ObjData, IndexTuple};

use super::{Vec3, vec3::Vec3u, Transform};

#[derive(Debug)]
pub enum MeshError {
    MeshLoadError(ObjError),
    MaterialLoadError(MtlLibsLoadError),
    UntriangulatedError,
}

impl From<ObjError> for MeshError {
    fn from(x: ObjError) -> Self {
        MeshError::MeshLoadError(x)
    }
}

impl From<MtlLibsLoadError> for MeshError {
    fn from(x: MtlLibsLoadError) -> Self {
        MeshError::MaterialLoadError(x)
    }
}

pub struct Mesh {
    pub(crate) vertices: Vec<Vec3>,
    pub(crate) tris: Vec<Vec3u>
}

impl Mesh {
    pub fn from_obj_file(filepath: &Path) -> Result<Mesh, MeshError> {
        let mut obj: Obj = Obj::load(filepath)?;
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut tris: Vec<Vec3u> = Vec::new();
        let obj_data: &ObjData = &obj.data;
        for chunk in obj_data.position.iter() {
            vertices.push(Vec3(chunk[0], chunk[1], chunk[2]));
        }

        let object = &obj_data.objects[0];
        for group in object.groups.iter() {
            for poly in group.polys.iter() {
                let index_tup_vec: &Vec<IndexTuple> = &poly.0;
                if index_tup_vec.len() != 3 {
                    return Err(MeshError::UntriangulatedError);
                }

                let v0 = index_tup_vec[0];
                let v1 = index_tup_vec[1];
                let v2 = index_tup_vec[2];
                tris.push(Vec3u(v0.0 as u32, v1.0 as u32, v2.0 as u32));
            }
        }

        Ok(
            Mesh { vertices: vertices, tris: tris }
        )
    }

    pub fn from_gltf_mesh(mesh: &gltf::Mesh, buffers: &[gltf::buffer::Data]) -> Mesh {
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut tris: Vec<Vec3u> = Vec::new();

        assert!(mesh.primitives().len() == 1);
        let primitive = mesh.primitives().next().unwrap();
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        if let Some(iter) = reader.read_positions() {
            for vertex in iter {
                vertices.push(Vec3(vertex[0], vertex[1], vertex[2]));
                // println!("point {} {} {}", vertex[0], vertex[1], vertex[2]);
            }
        }
        if let Some(iter) = reader.read_indices() {
            let mut tri_indices = [0; 3];
            for (i, tri) in iter.into_u32().enumerate() {
                tri_indices[i % 3] = tri;
                if i % 3 == 2 {
                    tris.push(Vec3u(tri_indices[0], tri_indices[1], tri_indices[2]));
                    // println!("tri {} {} {}", tri_indices[0], tri_indices[1], tri_indices[2]);
                }
            }
        }
        
        Mesh {
            vertices,
            tris,
        }
    }

    pub fn apply_transform(&mut self, transform: Transform) {
        for vertex in self.vertices.iter_mut() {
            let transformed = transform.apply_point(*vertex);
            *vertex = transformed;
        }
    }

}