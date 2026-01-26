use std::borrow::Cow;
use std::io::BufRead;
use std::path::Path;

use anyhow::{Context, Result};
use ply_rs_bw::parser::Parser;
use ply_rs_bw::ply::{Property, PropertyAccess};
use tracing::warn;

use crate::geometry::{vec3::Vec3u, Vec2, Vec3};

#[derive(Default)]
struct PlyVertex {
    x: f32,
    y: f32,
    z: f32,
    nx: Option<f32>,
    ny: Option<f32>,
    nz: Option<f32>,
    u: Option<f32>,
    v: Option<f32>,
}

impl PropertyAccess for PlyVertex {
    fn new() -> Self {
        PlyVertex::default()
    }

    fn set_property(&mut self, key: &str, property: Property) {
        match (key, property) {
            ("x", Property::Float(v)) => self.x = v,
            ("y", Property::Float(v)) => self.y = v,
            ("z", Property::Float(v)) => self.z = v,
            ("nx", Property::Float(v)) => self.nx = Some(v),
            ("ny", Property::Float(v)) => self.ny = Some(v),
            ("nz", Property::Float(v)) => self.nz = Some(v),
            ("s", Property::Float(v)) | ("u", Property::Float(v)) => self.u = Some(v),
            ("t", Property::Float(v)) | ("v", Property::Float(v)) => self.v = Some(v),
            _ => {}
        }
    }
}

#[derive(Default)]
struct PlyFace {
    vertex_indices: Vec<u32>,
}

impl PropertyAccess for PlyFace {
    fn new() -> Self {
        PlyFace::default()
    }

    fn set_property(&mut self, key: &str, property: Property) {
        match (key, property) {
            ("vertex_indices" | "vertex_index", Property::ListInt(v)) => {
                self.vertex_indices = v.into_iter().map(|i| i as u32).collect();
            }
            ("vertex_indices" | "vertex_index", Property::ListUInt(v)) => {
                self.vertex_indices = v;
            }
            _ => {}
        }
    }
}

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
    pub fn from_ply<P: AsRef<Path>>(path: P) -> Result<Mesh> {
        let file = std::fs::File::open(path.as_ref())
            .with_context(|| format!("failed to open PLY file: {:?}", path.as_ref()))?;
        let reader = std::io::BufReader::new(file);
        Self::from_ply_reader(reader)
    }

    pub fn from_ply_reader<R: BufRead>(mut reader: R) -> Result<Mesh> {
        let parser = Parser::<PlyVertex>::new();
        let header = parser.read_header(&mut reader).context("failed to parse PLY header")?;

        let mut ply_vertices: Vec<PlyVertex> = Vec::new();
        let mut ply_faces: Vec<PlyFace> = Vec::new();

        for (_, element) in &header.elements {
            match element.name.as_str() {
                "vertex" => {
                    ply_vertices = parser
                        .read_payload_for_element(&mut reader, element, &header)
                        .context("failed to read PLY vertices")?;
                }
                "face" => {
                    let face_parser = Parser::<PlyFace>::new();
                    ply_faces = face_parser
                        .read_payload_for_element(&mut reader, element, &header)
                        .context("failed to read PLY faces")?;
                }
                _ => {}
            }
        }

        let has_normals = ply_vertices.first().is_some_and(|v| v.nx.is_some());
        let has_uvs = ply_vertices.first().is_some_and(|v| v.u.is_some());

        let mut vertices = Vec::with_capacity(ply_vertices.len());
        let mut normals = Vec::with_capacity(if has_normals { ply_vertices.len() } else { 0 });
        let mut uvs = Vec::with_capacity(if has_uvs { ply_vertices.len() } else { 0 });

        for v in ply_vertices {
            vertices.push(Vec3(v.x, v.y, v.z));
            if has_normals {
                normals.push(Vec3(
                    v.nx.unwrap_or(0.0),
                    v.ny.unwrap_or(0.0),
                    v.nz.unwrap_or(0.0),
                ));
            }
            if has_uvs {
                uvs.push(Vec2(v.u.unwrap_or(0.0), v.v.unwrap_or(0.0)));
            }
        }

        let mut tris = Vec::new();
        for face in ply_faces {
            let indices = &face.vertex_indices;
            if indices.len() >= 3 {
                // Triangulate polygon using fan triangulation
                for i in 1..indices.len() - 1 {
                    tris.push(Vec3u(indices[0], indices[i] as u32, indices[i + 1] as u32));
                }
            }
        }

        Ok(Mesh {
            vertices,
            tris,
            normals,
            uvs,
        })
    }

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
                        let uv = [uv[0] as f32 / u8::MAX as f32, uv[1] as f32 / u8::MAX as f32];
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
