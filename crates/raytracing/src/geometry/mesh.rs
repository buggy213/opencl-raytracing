use std::path::Path;

use super::{Vec3, vec3::Vec3u, Transform};

#[derive(Debug)]
pub enum MeshError {
    UntriangulatedError,
}

#[derive(Debug)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub tris: Vec<Vec3u>,
    pub normals: Vec<Vec3>
}

impl Mesh {
    pub fn from_gltf_mesh(mesh: &gltf::Mesh, buffers: &[gltf::buffer::Data]) -> Mesh {
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut tris: Vec<Vec3u> = Vec::new();
        let mut normals: Vec<Vec3> = Vec::new();

        assert!(mesh.primitives().len() == 1, "Multiple materials not supported yet");

        let primitive = mesh.primitives().next().unwrap();
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
        
        if let Some(iter) = reader.read_positions() {
            for vertex in iter {
                vertices.push(Vec3(vertex[0], vertex[1], vertex[2]));
            }
        } else { todo!("unable to load vertices"); }
        
        if let Some(iter) = reader.read_indices() {
            let mut tri_indices = [0; 3];
            for (i, tri) in iter.into_u32().enumerate() {
                tri_indices[i % 3] = tri;
                if i % 3 == 2 {
                    tris.push(Vec3u(tri_indices[0], tri_indices[1], tri_indices[2]));
                }
            }
        } else { todo!("unable to load tris"); }

        if let Some(iter) = reader.read_normals() {
            for normal in iter {
                normals.push(Vec3(normal[0], normal[1], normal[2]));
            }
        } else { todo!("unable to load normals"); }
        
        Mesh {
            vertices,
            tris,
            normals
        }
    }

    pub fn apply_transform(&mut self, transform: &Transform) {
        for vertex in self.vertices.iter_mut() {
            let transformed = transform.apply_point(*vertex);
            *vertex = transformed;
        }

        for normal in self.normals.iter_mut() {
            let transformed = transform.apply_normal(*normal);
            *normal = transformed;
        }
    }

}