pub enum Material {
    Diffuse { 
        albedo: [f32; 3] 
    },
    Dielectric { 
        eta: f32 
    },
    Conductor { 
        eta: f32, 
        k: f32 
    },
}