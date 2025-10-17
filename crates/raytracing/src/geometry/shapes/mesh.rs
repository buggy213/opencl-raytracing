use crate::geometry::{Vec3, vec3::Vec3u};

// Note that a single "logical" mesh could be split into multiple Mesh structs
// if it contains >1 material. Each Mesh struct actually maps more closely to 
// "primitive" concept in GLTF. 
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vec3>,
    pub tris: Vec<Vec3u>,
    pub normals: Vec<Vec3>,
}

impl Mesh {
    pub fn from_gltf_primitive(primitive: gltf::Primitive, buffers: &[gltf::buffer::Data]) -> Mesh {
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut tris: Vec<Vec3u> = Vec::new();
        let mut normals: Vec<Vec3> = Vec::new();

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
                normals.push(normal.into());
            }
        } else { todo!("unable to load normals"); }
        
        Mesh {
            vertices,
            tris,
            normals,
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