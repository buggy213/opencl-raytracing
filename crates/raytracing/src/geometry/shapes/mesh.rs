use std::borrow::Cow;

use tracing::warn;

use crate::geometry::{vec3::Vec3u, Vec2, Vec3};

// Note that a single "logical" mesh could be split into multiple Mesh structs
// if it contains >1 material. Each Mesh struct actually maps more closely to
// "primitive" concept in GLTF.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub tris: Vec<Vec3u>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
}

impl Mesh {
    pub fn from_gltf_primitive(
        mesh: gltf::Mesh,
        primitive: gltf::Primitive,
        buffers: &[gltf::buffer::Data],
    ) -> Mesh {
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut tris: Vec<Vec3u> = Vec::new();
        let mut normals: Vec<Vec3> = Vec::new();
        let mut uvs: Vec<Vec2> = Vec::new();

        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

        if let Some(iter) = reader.read_positions() {
            vertices.reserve_exact(iter.len());
            for vertex in iter {
                vertices.push(Vec3(vertex[0], vertex[1], vertex[2]));
            }
        } else {
            todo!("unable to load vertices");
        }

        if let Some(iter) = reader.read_indices() {
            let iter = iter.into_u32();
            tris.reserve_exact(iter.len());
            let mut tri_indices = [0; 3];
            for (i, tri) in iter.enumerate() {
                tri_indices[i % 3] = tri;
                if i % 3 == 2 {
                    tris.push(Vec3u(tri_indices[0], tri_indices[1], tri_indices[2]));
                }
            }
        } else {
            todo!("unable to load tris");
        }

        if let Some(iter) = reader.read_normals() {
            normals.reserve_exact(iter.len());
            for normal in iter {
                normals.push(normal.into());
            }
        } else {
            // TODO: calculate vertex normals here? or perpendicular to intersected triangle?
            todo!("unable to load normals");
        }

        if let Some(iter) = reader.read_tex_coords(0) {
            uvs.reserve_exact(vertices.len());
            match iter {
                gltf::mesh::util::ReadTexCoords::U8(iter) => {
                    for uv in iter {
                        let uv = [uv[0] as f32 / u8::MAX as f32, uv[0] as f32 / u8::MAX as f32];
                        uvs.push(uv.into());
                    }
                }
                gltf::mesh::util::ReadTexCoords::U16(iter) => {
                    for uv in iter {
                        let uv = [
                            uv[0] as f32 / u16::MAX as f32,
                            uv[1] as f32 / u16::MAX as f32,
                        ];
                        uvs.push(uv.into());
                    }
                }
                gltf::mesh::util::ReadTexCoords::F32(iter) => {
                    for uv in iter {
                        uvs.push(uv.into());
                    }
                }
            }
        } else {
            let mesh_name = match mesh.name() {
                Some(name) => Cow::Borrowed(name),
                None => Cow::Owned(format!("{}", mesh.index())),
            };

            warn!(
                "no uvs loaded for gltf primitive {} (in mesh {})",
                primitive.index(),
                mesh_name
            );
        }

        Mesh {
            vertices,
            tris,
            normals,
            uvs,
        }
    }

    pub fn area(&self) -> f32 {
        let mut area = 0.0;
        for tri in 0..self.tris.len() {
            area += self.tri_area(tri);
        }

        area
    }

    pub fn tri_area(&self, tri_idx: usize) -> f32 {
        let tri = self.tris[tri_idx];
        let p0 = self.vertices[tri.0 as usize];
        let p1 = self.vertices[tri.1 as usize];
        let p2 = self.vertices[tri.2 as usize];

        Vec3::cross(p1 - p0, p2 - p0).length() / 2.0
    }
}
