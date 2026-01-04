// encapsulates sampler settings, independent of real implementation
#[derive(Debug, Clone)]
pub enum Sampler {
    Independent,
    Stratified {
        jitter: bool,
        x_strata: u32,
        y_strata: u32,
    },
}
