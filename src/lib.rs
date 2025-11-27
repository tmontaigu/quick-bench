use std::env::Args;
use std::fmt::Write;
use std::ops::{Add, Div, Sub};
use std::time::{Duration, Instant};

use crate::util::parse_args;
use crate::wall_time::{WallTime, WallTimeSample, WallTimeStats};

mod quick;
mod util;

pub use quick::QuickBencher;

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

#[derive(Clone, Copy, Debug)]
pub enum Throughput {
    Elements(u64),
}

#[derive(Copy, Clone, Debug)]
pub struct BencherConfig {
    // TODO use NonZero ?
    pub num_samples: usize,
    // TODO use NonZero ?
    pub iter_per_sample: usize,
    pub warmup_samples: usize,
    pub time_limit: Option<Duration>,
}

impl Default for BencherConfig {
    fn default() -> Self {
        Self {
            num_samples: 10,
            iter_per_sample: 10,
            warmup_samples: 5,
            time_limit: None,
        }
    }
}

#[derive(Debug)]
pub struct Statistics<Extra = ()> {
    pub wall_time: WallTimeStats,
    pub extra: Extra,
}

enum BenchMode {
    Micro,
    Macro,
}

impl BenchMode {
    const DEFAULT_LIMIT: Duration = Duration::from_secs(1);

    fn select_with_input<F, I, O>(f: F, input: I) -> Self
    where
        F: Fn(I) -> O,
    {
        let start = Instant::now();
        let _ = f(input);
        let one_call_duration = start.elapsed();

        if one_call_duration >= Self::DEFAULT_LIMIT {
            Self::Macro
        } else {
            Self::Micro
        }
    }
}

pub struct GenericBencher<M: Metric = NoExtraMetric> {
    pub config: BencherConfig,
    wall_time_samples: Vec<WallTimeSample>,
    extra_samples: M::SampleCollection,
    throughput: Option<Throughput>,
}

impl<M> GenericBencher<M>
where
    M: Metric,
{
    pub fn new(config: BencherConfig) -> Self {
        Self {
            config,
            wall_time_samples: vec![],
            extra_samples: M::SampleCollection::new(),
            throughput: None,
        }
    }

    pub fn throughput(&mut self, throughput: Throughput) {
        self.throughput = Some(throughput);
    }

    pub fn time_limit(&mut self, limit: Option<Duration>) {
        self.config.time_limit = limit;
    }

    // pub fn change_measurement<T>(&self) -> GenericBencher<T>
    // where
    //     T: Metric,
    // {
    //     GenericBencher {
    //         config: self.config,
    //         samples: T::SampleCollection::new(),
    //         throughput: self.throughput,
    //     }
    // }

    #[inline(never)]
    pub fn bench<F, O>(&mut self, f: F) -> Option<Statistics<M::Statistics>>
    where
        F: Fn() -> O,
    {
        // Here we trust and hope that the compiler is able to remove
        // the extra closure call, and optimize out most of the
        // `()` being used.
        //
        // If we find that its not the case then we would have to
        // to the inlining by hand
        self.bench_with_inputs_impl(|| (), |()| f())
    }

    #[inline(never)]
    pub fn bench_with_inputs<G, F, I, R>(
        &mut self,
        input_gen: G,
        f: F,
    ) -> Option<Statistics<M::Statistics>>
    where
        G: Fn() -> I,
        F: Fn(I) -> R,
    {
        self.bench_with_inputs_impl(input_gen, f)
    }

    #[inline(always)]
    fn bench_with_inputs_impl<G, F, I, R>(
        &mut self,
        input_gen: G,
        f: F,
    ) -> Option<Statistics<M::Statistics>>
    where
        G: Fn() -> I,
        F: Fn(I) -> R,
    {
        if self.config.num_samples == 0 {
            return None;
        }

        self.extra_samples.clear();
        self.wall_time_samples.clear();

        let mut inputs = Vec::with_capacity(self.config.iter_per_sample);
        let mut outputs = Vec::with_capacity(self.config.iter_per_sample);

        let input = input_gen();
        let (time_limit, iter_per_samples) = match BenchMode::select_with_input(&f, input) {
            BenchMode::Micro => {
                for _ in 0..self.config.warmup_samples {
                    for _ in 0..self.config.iter_per_sample {
                        inputs.push(input_gen());
                    }
                    let _ = self.sample_time_with_inputs(&f, &mut inputs, &mut outputs);
                    outputs.clear();
                    self.extra_samples.clear();
                    self.wall_time_samples.clear();
                }

                (
                    self.config.time_limit.unwrap_or(Duration::MAX),
                    self.config.iter_per_sample,
                )
            }
            BenchMode::Macro => (self.config.time_limit.unwrap_or(Duration::from_mins(2)), 1),
        };
        self.wall_time_samples.clear();
        self.extra_samples.clear();
        self.extra_samples.ensure_capacity(self.config.num_samples);
        self.wall_time_samples
            .ensure_capacity(self.config.num_samples);

        let start = Instant::now();
        for sample_index in 0..self.config.num_samples {
            for _ in 0..iter_per_samples {
                inputs.push(input_gen());
            }
            self.sample_time_with_inputs(&f, &mut inputs, &mut outputs);
            outputs.clear();

            if sample_index != self.config.num_samples - 1 {
                if start.elapsed() >= time_limit {
                    println!(
                        "Timit limit reached, limit: {time_limit:?}, elapsed: {:?}",
                        start.elapsed()
                    );
                    break;
                }
            }
        }
        let total_elapsed = start.elapsed();

        let stats = self.compute_statistics(iter_per_samples as u64);
        self.print_stats(&stats, total_elapsed);

        Some(stats)
    }

    #[inline]
    fn sample_time_with_inputs<I, O, F>(&mut self, f: &F, inputs: &mut Vec<I>, outputs: &mut Vec<O>)
    where
        F: Fn(I) -> O,
    {
        let wt_start = WallTime::start();
        let extra_start = M::start();
        for input in inputs.drain(..) {
            let o = f(input);
            outputs.push(o);
        }
        let wt_sample = WallTime::end(&wt_start);
        let extra_sample = M::end(&extra_start);

        self.extra_samples.push(extra_sample);
        self.wall_time_samples.push(wt_sample);
    }

    fn compute_statistics(&self, iter_per_samples: u64) -> Statistics<M::Statistics> {
        let wt_stats = WallTimeStats::compute(
            &self.wall_time_samples,
            iter_per_samples as usize,
            self.throughput,
        );
        let extra_stats = M::compute_statistics(
            &self.extra_samples,
            &self.wall_time_samples,
            iter_per_samples as u64,
        );

        Statistics {
            wall_time: wt_stats,
            extra: extra_stats,
        }
    }

    fn print_stats(&self, stats: &Statistics<M::Statistics>, total_elapsed: Duration) {
        println!("\tRunning time: {total_elapsed:?}");
        WallTime::print_stats(&stats.wall_time);
        M::print_stats(&stats.extra);
        println!();
    }
}

pub type Bencher = GenericBencher<NoExtraMetric>;

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

pub struct Runner<M = NoExtraMetric>
where
    M: Metric,
{
    filter: Option<regex::Regex>,
    default_config: BencherConfig,
    bencher: GenericBencher<M>,
}

impl<M> Runner<M>
where
    M: Metric,
{
    pub fn new() -> Self {
        let default_config = BencherConfig::default();
        Self {
            filter: None,
            bencher: GenericBencher::new(default_config),
            default_config,
        }
    }

    pub fn setup_from_cmdline(self) -> Self {
        let mut args = std::env::args();
        self.setup_from_args(&mut args)
    }

    pub fn setup_from_args(mut self, args: &mut Args) -> Self {
        let config = parse_args(args);
        self.filter = config.filter;
        self.default_config = config.bencher_config;
        self
    }

    pub fn run(&mut self, id: &str, mut callback: impl FnMut(&mut GenericBencher<M>)) {
        let should_run = self.filter.as_ref().is_none_or(|regex| regex.is_match(id));

        if should_run {
            println!("{id} : Running");
            self.bencher.config = self.default_config;
            callback(&mut self.bencher);
        } else {
            println!("{id}: Skipped");
        }
    }

    pub fn run_with<B, A>(&mut self, mut benchable: B, args_iter: impl IntoIterator<Item = A>)
    where
        B: Benchable<A, M>,
    {
        let mut id = String::new();
        for args in args_iter {
            benchable.write_name(&args, &mut id);
            let should_run = self.filter.as_ref().is_none_or(|regex| regex.is_match(&id));

            if should_run {
                println!("{id} : Running");
                self.bencher.config = self.default_config;
                benchable.execute(&mut self.bencher, args);
            } else {
                println!("{id}: Skipped");
            }
            id.clear();
        }
    }
}

pub trait Benchable<A, M>
where
    M: Metric,
{
    fn write_name(&self, args: &A, name: &mut String);

    fn execute(&mut self, bencher: &mut GenericBencher<M>, args: A);
}

impl<F, A, M> Benchable<A, M> for (&str, F)
where
    F: Fn(&mut GenericBencher<M>, A),
    A: std::fmt::Debug + 'static,
    M: Metric,
{
    fn write_name(&self, args: &A, name: &mut String) {
        use std::any::TypeId;
        if TypeId::of::<A>() == TypeId::of::<()>() {
            write!(name, "{}", self.0).unwrap();
        } else {
            write!(name, "{}({args:?})", self.0).unwrap();
        }
    }

    fn execute(&mut self, bencher: &mut GenericBencher<M>, args: A) {
        self.1(bencher, args);
    }
}

impl Default for Runner {
    fn default() -> Self {
        Runner::<NoExtraMetric>::new()
    }
}

fn ensure_vec_capacity<T>(vec: &mut Vec<T>, cap: usize) {
    if cap > vec.capacity() {
        vec.reserve(cap - vec.capacity());
    }
}

pub trait SampleCollection: Sized {
    type Sample;

    fn clear(&mut self);

    fn new() -> Self;

    fn ensure_capacity(&mut self, cap: usize);

    fn push(&mut self, value: Self::Sample);
}

impl<T> SampleCollection for Vec<T> {
    type Sample = T;

    fn clear(&mut self) {
        self.clear();
    }

    fn new() -> Self {
        Self::new()
    }

    fn ensure_capacity(&mut self, cap: usize) {
        ensure_vec_capacity(self, cap);
    }

    fn push(&mut self, value: Self::Sample) {
        self.push(value);
    }
}

pub trait Metric {
    type Begin;
    type Sample;
    type SampleCollection: SampleCollection<Sample = Self::Sample>;

    type Statistics;

    fn start() -> Self::Begin;

    fn end(intermediate: &Self::Begin) -> Self::Sample;

    fn compute_statistics(
        samples: &Self::SampleCollection,
        wall_time_samples: &Vec<WallTimeSample>,
        iters_per_sample: u64,
    ) -> Self::Statistics;

    fn print_stats(stats: &Self::Statistics);
}

pub struct MultiSample2<A, B>
where
    A: Metric,
    B: Metric,
{
    a: A::SampleCollection,
    b: B::SampleCollection,
}

impl<A, B> SampleCollection for MultiSample2<A, B>
where
    A: Metric,
    B: Metric,
{
    type Sample = (A::Sample, B::Sample);

    fn clear(&mut self) {
        self.a.clear();
        self.b.clear();
    }

    fn new() -> Self {
        Self {
            a: A::SampleCollection::new(),
            b: B::SampleCollection::new(),
        }
    }

    fn push(&mut self, (a, b): Self::Sample) {
        self.a.push(a);
        self.b.push(b);
    }

    fn ensure_capacity(&mut self, cap: usize) {
        self.a.ensure_capacity(cap);
        self.b.ensure_capacity(cap);
    }
}

impl<A, B> Default for MultiSample2<A, B>
where
    A: Metric,
    B: Metric,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<A, B> Metric for (A, B)
where
    A: Metric,
    B: Metric,
{
    type Begin = (A::Begin, B::Begin);

    type Sample = (A::Sample, B::Sample);

    type SampleCollection = MultiSample2<A, B>;

    type Statistics = (A::Statistics, B::Statistics);

    fn start() -> Self::Begin {
        (A::start(), B::start())
    }

    fn end(intermediate: &Self::Begin) -> Self::Sample {
        (A::end(&intermediate.0), B::end(&intermediate.1))
    }

    fn compute_statistics(
        samples: &Self::SampleCollection,
        wt_stats: &Vec<WallTimeSample>,
        iters_per_sample: u64,
    ) -> Self::Statistics {
        (
            A::compute_statistics(&samples.a, wt_stats, iters_per_sample),
            B::compute_statistics(&samples.b, wt_stats, iters_per_sample),
        )
    }

    fn print_stats((a, b): &Self::Statistics) {
        A::print_stats(a);
        B::print_stats(b);
    }
}

pub struct NoExtraMetric;
pub struct NaughtCollection;

impl SampleCollection for NaughtCollection {
    type Sample = ();

    fn clear(&mut self) {
        ()
    }

    fn new() -> Self {
        NaughtCollection
    }

    fn ensure_capacity(&mut self, _cap: usize) {
        ()
    }

    fn push(&mut self, _value: Self::Sample) {
        ()
    }
}

impl Metric for NoExtraMetric {
    type Begin = ();

    type Sample = ();

    type SampleCollection = NaughtCollection;

    type Statistics = ();

    fn start() -> Self::Begin {
        ()
    }

    fn end(_intermediate: &Self::Begin) -> Self::Sample {
        ()
    }

    fn compute_statistics(
        _samples: &Self::SampleCollection,
        _wt_samples: &Vec<WallTimeSample>,
        _iters_per_sample: u64,
    ) -> Self::Statistics {
        ()
    }

    fn print_stats(_stats: &Self::Statistics) {
        ()
    }
}

mod wall_time {
    use std::{
        ops::{Add, Div, Sub},
        time::Duration,
    };

    use crate::{Metric, SaturatingSub, Throughput};

    #[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Debug)]
    pub struct WallTimeSample {
        pub(crate) duration: Duration,
    }

    impl Add<Self> for WallTimeSample {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self {
                duration: self.duration + rhs.duration,
            }
        }
    }

    impl Sub<Self> for WallTimeSample {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self {
                duration: self.duration - rhs.duration,
            }
        }
    }

    impl SaturatingSub<Self> for WallTimeSample {
        type Output = Self;

        fn saturating_sub(self, rhs: Self) -> Self::Output {
            Self {
                duration: self.duration.saturating_sub(rhs.duration),
            }
        }
    }

    impl Div<u32> for WallTimeSample {
        type Output = Self;

        fn div(self, rhs: u32) -> Self::Output {
            Self {
                duration: self.duration / rhs,
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct WallTimeStats {
        pub(crate) num_outliers: usize,
        pub(crate) min: Duration,
        pub(crate) max: Duration,
        pub(crate) avg: Duration,
        pub(crate) std_dev: Duration,
        pub(crate) throughput: Option<f64>,
    }

    impl WallTimeStats {
        pub(crate) fn compute(
            samples: &[WallTimeSample],
            iter_per_sample: usize,
            throughput: Option<Throughput>,
        ) -> Self {
            // TODO: remove to_vec
            let (mut inliers, outliers) = super::split_outliers(samples.to_vec());

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

            let throughput = throughput.map(|t| match t {
                Throughput::Elements(n) => {
                    let total = n * iter_per_sample as u64 * inliers.len() as u64;
                    let elem_per_secs = total as f64 / sum.as_secs_f64();
                    elem_per_secs
                }
            });

            Self {
                num_outliers: outliers.len(),
                min,
                max,
                avg: average,
                std_dev: stddev,
                throughput,
            }
        }
    }

    pub struct WallTime;

    impl Metric for WallTime {
        type Begin = std::time::Instant;
        type Sample = WallTimeSample;
        type SampleCollection = Vec<Self::Sample>;
        type Statistics = WallTimeStats;

        fn start() -> Self::Begin {
            std::time::Instant::now()
        }

        fn end(intermediate: &Self::Begin) -> Self::Sample {
            WallTimeSample {
                duration: intermediate.elapsed(),
            }
        }

        fn compute_statistics(
            samples: &Self::SampleCollection,
            _wt_samples: &Vec<WallTimeSample>,
            iters_per_sample: u64,
        ) -> Self::Statistics {
            WallTimeStats::compute(samples, iters_per_sample as usize, None)
        }

        fn print_stats(stats: &Self::Statistics) {
            let WallTimeStats {
                num_outliers,
                min,
                max,
                avg,
                std_dev: stddev,
                throughput,
            } = stats;

            println!("\tWallTime:");
            println!("\t\tOutliers: {}", num_outliers);
            println!("\t\taverage: {avg:?}, min: {min:?}, max: {max:?}, stddev: {stddev:?}");

            if let Some(t) = throughput {
                println!("\tThroughput: {t} e/s");
            }
        }
    }
}

trait SaturatingSub<Rhs = Self> {
    type Output;

    fn saturating_sub(self, rhs: Rhs) -> Self::Output;
}

fn split_outliers<T>(input: Vec<T>) -> (Vec<T>, Vec<T>)
where
    T: Copy
        + Ord
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + SaturatingSub<T, Output = T>
        + Div<u32, Output = T>,
{
    if input.len() < 4 {
        return (input, vec![]);
    }

    let mut sorted = input;
    sorted.sort();

    let q1_idx = sorted.len() / 4;
    let q3_idx = (3 * sorted.len()) / 4;
    let q1 = sorted[q1_idx];
    let q3 = sorted[q3_idx];
    let iqr = q3 - q1;

    let step = iqr + iqr / 2; // 1.5 * IQR
    let lower_bound = q1.saturating_sub(step); // Q1 - 1.5*IQR
    let upper_bound = q3 + step; // Q3 + 1.5*IQR

    let mut outliers = vec![];
    let mut inliers = vec![];
    for stat in sorted {
        if stat < lower_bound || stat > upper_bound {
            outliers.push(stat);
        } else {
            inliers.push(stat);
        }
    }

    (inliers, outliers)
}

fn split_outliers_by_index<T>(input: &[T]) -> (Vec<usize>, Vec<usize>)
where
    T: Copy
        + Ord
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + SaturatingSub<T, Output = T>
        + Div<u32, Output = T>,
{
    if input.len() < 4 {
        return (vec![0, 1, 2], vec![]);
    }

    let mut sorted = input.to_vec();
    sorted.sort();

    let q1_idx = sorted.len() / 4;
    let q3_idx = (3 * sorted.len()) / 4;
    let q1 = sorted[q1_idx];
    let q3 = sorted[q3_idx];
    let iqr = q3 - q1;

    let step = iqr + iqr / 2; // 1.5 * IQR
    let lower_bound = q1.saturating_sub(step); // Q1 - 1.5*IQR
    let upper_bound = q3 + step; // Q3 + 1.5*IQR

    let mut outliers = vec![];
    let mut inliers = vec![];
    for stat in sorted {
        let index = input.iter().position(|&v| v == stat).unwrap();
        if stat < lower_bound || stat > upper_bound {
            outliers.push(index);
        } else {
            inliers.push(index);
        }
    }

    (inliers, outliers)
}
#[cfg(feature = "libc")]
pub mod cpu_time {
    use std::{
        mem::MaybeUninit,
        ops::{Add, Div, Sub},
        time::Duration,
    };

    use libc::timespec;

    use crate::{Metric, SaturatingSub};

    pub struct CpuLoadStatistics {
        pub mean_percent: f64,
    }

    pub struct CpuTimeInstant(timespec);

    impl CpuTimeInstant {
        pub fn now() -> Self {
            let inner = unsafe {
                let mut tp = MaybeUninit::<timespec>::uninit();
                let r = libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, tp.as_mut_ptr());
                assert_eq!(r, 0);
                tp.assume_init()
            };

            Self(inner)
        }

        pub fn elapsed(&self) -> CpuTimeDuration {
            let now = Self::now();

            let sec = now.0.tv_sec - self.0.tv_sec;
            let nsec = now.0.tv_nsec - self.0.tv_nsec;

            CpuTimeDuration(Duration::new(sec as u64, nsec as u32))
        }
    }

    #[derive(Debug, Clone, Copy, Ord, PartialOrd, PartialEq, Eq)]
    pub struct CpuTimeDuration(Duration);

    impl Add<Self> for CpuTimeDuration {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    impl Sub<Self> for CpuTimeDuration {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 - rhs.0)
        }
    }

    impl SaturatingSub<Self> for CpuTimeDuration {
        type Output = Self;

        fn saturating_sub(self, rhs: Self) -> Self::Output {
            Self(self.0.saturating_sub(rhs.0))
        }
    }

    impl Div<u32> for CpuTimeDuration {
        type Output = Self;

        fn div(self, rhs: u32) -> Self::Output {
            Self(self.0 / rhs)
        }
    }

    impl CpuTimeDuration {
        pub fn percent(&self, thread_count: usize, wall_time: Duration) -> f64 {
            let mut elapsed = self.0.as_nanos() as f64;
            elapsed /= wall_time.as_nanos() as f64;
            elapsed /= thread_count as f64;
            elapsed * 100.0
        }
    }

    pub struct CpuLoad;

    impl Metric for CpuLoad {
        type Begin = CpuTimeInstant;

        type Sample = CpuTimeDuration;

        type SampleCollection = Vec<Self::Sample>;

        type Statistics = CpuLoadStatistics;

        fn start() -> Self::Begin {
            CpuTimeInstant::now()
        }

        fn end(intermediate: &Self::Begin) -> Self::Sample {
            intermediate.elapsed()
        }

        fn compute_statistics(
            samples: &Self::SampleCollection,
            wall_time_samples: &Vec<crate::wall_time::WallTimeSample>,
            _iters_per_sample: u64,
        ) -> Self::Statistics {
            let threads = std::thread::available_parallelism()
                .expect("Failed to get CPU threads")
                .get();

            let (inliers, _outliers) = super::split_outliers_by_index(&samples);

            let mut sum = 0.0;
            for index in inliers {
                let sample = samples[index];
                let wall_time = wall_time_samples[index];
                let p = sample.percent(threads, wall_time.duration);
                sum += p;
            }

            CpuLoadStatistics {
                mean_percent: sum / (wall_time_samples.len() as f64),
            }
        }

        fn print_stats(stats: &Self::Statistics) {
            println!("\tCpuLoad: Average: {:.03}%", stats.mean_percent);
        }
    }
}
