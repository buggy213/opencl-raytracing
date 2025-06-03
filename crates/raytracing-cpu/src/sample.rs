use raytracing::geometry::Vec2;

pub(crate) fn sample_uniform() -> f32 {
    rand::random_range(0.0 .. 1.0)
}

pub(crate) fn sample_uniform2() -> Vec2 {
    Vec2(sample_uniform(), sample_uniform())
}