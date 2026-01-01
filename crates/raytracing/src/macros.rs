macro_rules! variadic_min {
    ($x:expr) => ($x);
    ($x:expr, $($y:expr),+) => (
        std::cmp::min($x, crate::macros::variadic_min!($($y),+))
    );
}

macro_rules! variadic_max {
    ($x:expr) => ($x);
    ($x:expr, $($y:expr),+) => (
        std::cmp::max($x, crate::macros::variadic_min!($($y),+))
    );
}

macro_rules! variadic_min_comparator {
    ($min_cmp:path, $x:expr) => ($x);
    ($min_cmp:path, $x:expr, $($y:expr),+) => (
        $min_cmp($x, crate::macros::variadic_min_comparator!($min_cmp, $($y),+))
    );
}

macro_rules! variadic_max_comparator {
    ($max_cmp:path, $x:expr) => ($x);
    ($max_cmp:path, $x:expr, $($y:expr),+) => (
        $max_cmp($x, crate::macros::variadic_max_comparator!($max_cmp, $($y),+))
    );
}

pub(crate) use variadic_max;
pub(crate) use variadic_max_comparator;
pub(crate) use variadic_min;
pub(crate) use variadic_min_comparator;
