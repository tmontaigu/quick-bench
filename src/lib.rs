use std::borrow::Cow;
use std::env::Args;
use std::mem::MaybeUninit;
use std::time::Duration;

#[derive(Default)]
pub struct OrchestratorConfig {
    filter: Option<regex::Regex>,
    bencher_config: BencherConfig,
}

// Feature List:
// * Use time for warmup, i.e., warmup for x seconds
//   to allow user to then parse its own args. (or use `--` as separator)
// * override bencher settings
// * write measurements to file;
//
// * QuickBencher type
//   * with a measure<F, R>(f: F) -> R
//   * user call print stats at the end
//

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
    samples: Vec<Sample>,
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
        let mut output = SingleOutputContainer::<O>::new();
        let duration = sample_time(&func, Self::ITER_PER_SAMPLE as u64, &mut output);
        self.samples.push(Sample { duration });
        output.unwrap()
    }

    pub fn compute_and_print_stats(&self) {
        if self.samples.is_empty() {
            println!("No samples");
            return;
        }

        let SampleStats {
            num_outliers: outliers,
            min,
            max,
            avg,
            std_dev: stddev,
        } = SampleStats::compute(self.samples.as_slice(), Self::ITER_PER_SAMPLE);
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

#[derive(Copy, Clone, Debug)]
pub struct BencherConfig {
    num_samples: usize,
    iter_per_sample: usize,
    warmup_samples: usize,
}

impl Default for BencherConfig {
    fn default() -> Self {
        Self {
            num_samples: 10,
            iter_per_sample: 10,
            warmup_samples: 5,
        }
    }
}

fn parse_args(args: &mut Args) -> OrchestratorConfig {
    let iter = args.by_ref();
    let mut config = OrchestratorConfig::default();

    // Skip first arg, which is the binary name
    if iter.next().is_none() {
        return config;
    }

    // Then '--' (at least when invoked from cargo bench)
    if iter.next().is_none() {
        return config;
    }

    loop {
        let Some(arg) = iter.next() else {
            break;
        };

        if arg == "--" {
            break;
        }

        if arg == "--bench" {
            continue;
        }

        if let Some(long_arg) = arg.strip_prefix("--") {
            let (key_str, value_str) = if long_arg.contains("=") {
                // The arg is key=value
                let mut split = long_arg.splitn(2, '=');
                (
                    Cow::Borrowed(split.next().unwrap()),
                    Cow::Borrowed(split.next().unwrap()),
                )
            } else {
                // The value is on the next arg
                let key_str = long_arg;
                let value_str = iter.next().unwrap();
                (Cow::Borrowed(key_str), Cow::Owned(value_str))
            };

            if key_str == "num-samples" {
                config.bencher_config.num_samples = value_str.parse::<usize>().unwrap();
            } else if key_str == "iter-per-sample" {
                config.bencher_config.iter_per_sample = value_str.parse::<usize>().unwrap();
            } else if key_str == "warmup-samples" {
                config.bencher_config.warmup_samples = value_str.parse::<usize>().unwrap();
            } else {
                println!("Unknown argument: {}", arg);
            }
        } else {
            config.filter = Some(regex::Regex::new(&arg).unwrap());
        }
    }

    config
}

#[derive(Clone, Ord, PartialOrd, Eq, PartialEq)]
struct Sample {
    duration: Duration,
}

struct SampleStats {
    num_outliers: usize,
    min: Duration,
    max: Duration,
    avg: Duration,
    std_dev: Duration,
}

impl SampleStats {
    fn compute(samples: &[Sample], iter_per_sample: usize) -> Self {
        let (mut inliers, outliers) = split_outliers(samples);

        let mut sum = inliers[0].duration;
        let mut min = inliers[0].duration;
        let mut max = inliers[0].duration;
        for (i, stat) in inliers[1..].iter().enumerate() {
            if let Some(s) = sum.checked_add(stat.duration) {
                sum = s;
            } else {
                inliers.truncate(i + 1);
                println!(
                    "Warning: overflow in samples, ignoring sample above {}",
                    i + 1
                );
                break;
            }
            max = max.max(stat.duration);
            min = min.min(stat.duration);
        }
        let sampled_average = sum / inliers.len() as u32;
        let sampled_average_nano = sampled_average.as_nanos() as i128;

        let variance_nano = if inliers.len() <= 1 {
            0
        } else {
            inliers
                .iter()
                .map(|stat| {
                    let deviation = (stat.duration.as_nanos() as i128) - sampled_average_nano;
                    deviation * deviation
                })
                .sum::<i128>()
                / (inliers.len() as i128 - 1)
        };

        let stddev = Duration::from_nanos(variance_nano.isqrt() as u64);

        let average = sampled_average / iter_per_sample as u32;
        min /= iter_per_sample as u32;
        max /= iter_per_sample as u32;

        Self {
            num_outliers: outliers.len(),
            min,
            max,
            avg: average,
            std_dev: stddev,
        }
    }
}

pub struct Bencher {
    config: BencherConfig,
    samples: Vec<Sample>,
}

impl Bencher {
    pub fn new(config: BencherConfig) -> Self {
        let stats = Vec::with_capacity(config.num_samples);
        Self {
            config,
            samples: stats,
        }
    }

    pub fn config(&self) -> &BencherConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut BencherConfig {
        &mut self.config
    }

    pub fn bench<F, O>(&mut self, f: F)
    where
        F: Fn() -> O,
    {
        if self.config.num_samples == 0 {
            return;
        }

        self.samples.clear();
        let mut outputs = Vec::with_capacity(self.config.iter_per_sample);
        for _ in 0..self.config.warmup_samples {
            let _ = sample_time(&f, self.config.iter_per_sample as u64, &mut outputs);
            outputs.clear();
        }

        for _ in 0..self.config.num_samples {
            let duration = sample_time(&f, self.config.iter_per_sample as u64, &mut outputs);
            outputs.clear();
            self.samples.push(Sample { duration });
        }

        self.compute_and_print_stats();
    }

    pub fn bench_with_inputs<G, F, I, R>(&mut self, input_gen: G, f: F)
    where
        G: Fn() -> I,
        F: Fn(I) -> R,
    {
        if self.config.num_samples == 0 {
            return;
        }

        self.samples.clear();
        let mut inputs = Vec::with_capacity(self.config.iter_per_sample);
        let mut outputs = Vec::with_capacity(self.config.iter_per_sample);

        for _ in 0..self.config.warmup_samples {
            for _ in 0..self.config.iter_per_sample {
                inputs.push(input_gen());
            }
            let _ = sample_time_with_inputs(&f, &mut inputs, &mut outputs);
            outputs.clear();
        }

        for _ in 0..self.config.num_samples {
            for _ in 0..self.config.iter_per_sample {
                inputs.push(input_gen());
            }
            let duration = sample_time_with_inputs(&f, &mut inputs, &mut outputs);
            self.samples.push(Sample { duration });
            outputs.clear();
        }

        self.compute_and_print_stats();
    }

    fn compute_and_print_stats(&self) {
        let SampleStats {
            num_outliers: outliers,
            min,
            max,
            avg,
            std_dev: stddev,
        } = SampleStats::compute(self.samples.as_slice(), self.config.iter_per_sample);
        println!("\tSamples: {}, Outliers: {}", self.samples.len(), outliers);
        println!("\tAverage: {avg:?}, min: {min:?}, max: {max:?}, stddev: {stddev:?}");
        println!();
    }
}

pub type BoxedBench = Box<dyn FnMut(&'_ mut Bencher)>;

pub struct BenchStore {
    vec: Vec<(String, BoxedBench)>,
    filter: Option<regex::Regex>,
    default_config: BencherConfig,
    bencher: Bencher,
}

impl BenchStore {
    pub fn new() -> Self {
        Self {
            vec: vec![],
            filter: None,
            bencher: Bencher::new(BencherConfig::default()),
            default_config: BencherConfig::default(),
        }
    }

    pub fn setup_from_cmdline() -> Self {
        let mut args = std::env::args();
        Self::setup_from_args(&mut args)
    }

    pub fn setup_from_args(args: &mut Args) -> Self {
        let config = parse_args(args);

        Self {
            vec: vec![],
            filter: config.filter,
            bencher: Bencher::new(config.bencher_config),
            default_config: config.bencher_config,
        }
    }

    pub fn with_filter(filter: &str) -> Self {
        Self {
            vec: vec![],
            filter: Some(regex::Regex::new(filter).unwrap()),
            bencher: Bencher::new(BencherConfig::default()),
            default_config: BencherConfig::default(),
        }
    }

    pub fn register(&mut self, id: String, callback: impl FnMut(&'_ mut Bencher) + 'static) {
        self.vec.push((id, Box::new(callback)));
    }

    pub fn run(mut self) {
        // or take &mut self, filter by draining into another vec
        // then at the end re-push into the same vec.
        // this would allow to call `run` multiple times.
        let original_count = self.vec.len();
        let mut benches_to_run = if let Some(ref filter) = self.filter {
            let mut benches_to_run = vec![];
            for (id, func) in self.vec.into_iter() {
                if !filter.is_match(&id) {
                    println!("{id}: Skipped");
                    continue;
                }
                benches_to_run.push((id, func));
            }
            benches_to_run
        } else {
            self.vec
        };

        println!(
            "Running {} / {original_count} benchmarks",
            benches_to_run.len()
        );
        let len = benches_to_run.len();
        for (i, (id, callback)) in benches_to_run.iter_mut().enumerate() {
            println!("[{} / {len}] {id} : Running", i + 1);
            self.bencher.config = self.default_config;
            callback(&mut self.bencher);
        }
    }
}

impl Default for BenchStore {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Runner {
    filter: Option<regex::Regex>,
    default_config: BencherConfig,
    bencher: Bencher,
}

impl Runner {
    pub fn new() -> Self {
        let default_config = BencherConfig::default();
        Self {
            filter: None,
            bencher: Bencher::new(default_config),
            default_config,
        }
    }

    pub fn setup_from_cmdline() -> Self {
        let mut args = std::env::args();
        Self::setup_from_args(&mut args)
    }

    pub fn setup_from_args(args: &mut Args) -> Self {
        let config = parse_args(args);

        Self {
            filter: config.filter,
            bencher: Bencher::new(config.bencher_config),
            default_config: config.bencher_config,
        }
    }

    pub fn run(&mut self, id: &str, mut callback: impl FnMut(&mut Bencher)) {
        let should_run = self.filter.as_ref().is_none_or(|regex| regex.is_match(id));

        if should_run {
            println!("{id} : Running");
            self.bencher.config = self.default_config;
            callback(&mut self.bencher);
        } else {
            println!("{id}: Skipped");
        }
    }
}

impl Default for Runner {
    fn default() -> Self {
        Self::new()
    }
}

pub trait OutputContainer<T> {
    fn push(&mut self, value: T);
}

impl<T> OutputContainer<T> for Vec<T> {
    fn push(&mut self, value: T) {
        self.push(value);
    }
}

struct SingleOutputContainer<T> {
    output: MaybeUninit<T>,
    is_init: bool,
}

impl<T> SingleOutputContainer<T> {
    fn new() -> Self {
        Self {
            output: MaybeUninit::uninit(),
            is_init: false,
        }
    }

    fn unwrap(mut self) -> T {
        self.replace().expect("The value is not initialized")
    }

    fn replace(&mut self) -> Option<T> {
        if self.is_init {
            let old = std::mem::replace(&mut self.output, MaybeUninit::uninit());
            self.is_init = false;
            unsafe {
                // SAFETY:
                // The invariant is that when is_init is true, the data is init
                Some(old.assume_init())
            }
        } else {
            None
        }
    }
}

impl<T> OutputContainer<T> for SingleOutputContainer<T> {
    fn push(&mut self, value: T) {
        if self.is_init {
            unsafe {
                // SAFETY:
                // The invariant is that when is_init is true, the data is init
                self.output.assume_init_drop();
            }
        }
        self.output.write(value);
        self.is_init = true;
    }
}

impl<T> Drop for SingleOutputContainer<T> {
    fn drop(&mut self) {
        if self.is_init {
            unsafe {
                // SAFETY:
                // The invariant is that when is_init is true, the data is init
                self.output.assume_init_drop();
            }
            // No need to reset is_init as we are getting dropped
        }
    }
}

#[inline]
fn sample_time<F, OutCont, O>(f: &F, iters_per_sample: u64, outputs: &mut OutCont) -> Duration
where
    OutCont: OutputContainer<O>,
    F: Fn() -> O,
{
    let now = std::time::Instant::now();
    for _ in 0..iters_per_sample {
        let o = f();
        outputs.push(o);
    }
    now.elapsed()
}

#[inline]
fn sample_time_with_inputs<I, O, F>(f: &F, inputs: &mut Vec<I>, outputs: &mut Vec<O>) -> Duration
where
    F: Fn(I) -> O,
{
    let now = std::time::Instant::now();
    for input in inputs.drain(..) {
        let o = f(input);
        outputs.push(o);
    }
    now.elapsed()
}

fn split_outliers(input: &[Sample]) -> (Vec<Sample>, Vec<Sample>) {
    if input.len() < 4 {
        return (input.to_vec(), vec![]);
    }

    let mut sorted = input.to_vec();
    sorted.sort();

    let q1_idx = sorted.len() / 4;
    let q3_idx = (3 * sorted.len()) / 4;
    let q1 = sorted[q1_idx].duration;
    let q3 = sorted[q3_idx].duration;
    let iqr = q3 - q1;

    let step = iqr + iqr / 2; // 1.5 * IQR
    let lower_bound = q1.saturating_sub(step); // Q1 - 1.5*IQR
    let upper_bound = q3 + step; // Q3 + 1.5*IQR

    let mut outliers = vec![];
    let mut inliers = vec![];
    for stat in sorted {
        if stat.duration < lower_bound || stat.duration > upper_bound {
            outliers.push(stat);
        } else {
            inliers.push(stat);
        }
    }

    (inliers, outliers)
}
