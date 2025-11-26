use crate::{WallTimeSample, WallTimeStats};

/// Simple struct to do quick 'benchmark'
///
/// This is a simple bencher where the caller controls when the
/// samples are measured. There is not warmup phase.
///
///
/// # Example
/// ```
/// # fn fibonacci(n: i32) -> i32 {
/// #     if n <= 0 {
/// #         0
/// #     } else if n == 1 {
/// #         1
/// #     } else {
/// #         fibonacci(n - 1) + fibonacci(n - 2)
/// #     }
/// # }
/// use quick_bench::QuickBencher;
///
/// let mut bencher = QuickBencher::new();
///
/// for _ in 0..10 {
///     let fib_result = bencher.sample_once(|| fibonacci(14));
///     assert_eq!(fib_result, 377);
/// }
///
/// bencher.compute_and_print_stats();
/// ```
pub struct QuickBencher {
    samples: Vec<WallTimeSample>,
}

impl QuickBencher {
    const ITER_PER_SAMPLE: usize = 1;

    pub fn new() -> Self {
        Self { samples: vec![] }
    }

    pub fn clear_samples(&mut self) {
        self.samples.clear();
    }

    pub fn sample_once<F, O>(&mut self, func: F) -> O
    where
        F: Fn() -> O,
    {
        let now = std::time::Instant::now();
        let o = func();
        self.samples.push(WallTimeSample {
            duration: now.elapsed(),
        });
        o
    }

    pub fn compute_and_print_stats(&self) {
        if self.samples.is_empty() {
            println!("No samples");
            return;
        }

        let WallTimeStats {
            num_outliers: outliers,
            min,
            max,
            avg,
            std_dev: stddev,
            throughput: _,
        } = WallTimeStats::compute(self.samples.as_slice(), Self::ITER_PER_SAMPLE, None);
        println!("Samples: {}, Outliers: {}", self.samples.len(), outliers);
        println!("Average: {avg:?}, min: {min:?}, max: {max:?}, stddev: {stddev:?}");
        println!();
    }
}

impl Default for QuickBencher {
    fn default() -> Self {
        Self::new()
    }
}
