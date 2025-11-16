use quick_bench::Runner;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn create_unsorted_vec(n: usize) -> Vec<u32> {
    rand::random_iter().take(n).collect()
}

fn fibonacci_recursive_if(n: i32) -> i32 {
    if n <= 0 {
        0
    } else if n == 1 {
        1
    } else {
        fibonacci_recursive_if(n - 1) + fibonacci_recursive_if(n - 2)
    }
}

fn fibonacci_recursive_match(term: u32) -> u32 {
    match term {
        0 | 1 => term,
        _ => fibonacci_recursive_match(term - 1) + fibonacci_recursive_match(term - 2), // Recursive calls
    }
}

fn fibonacci_iterative(term: u32) -> u32 {
    if term == 0 {
        return 0;
    }
    let (mut a, mut b) = (0, 1);
    for _ in 1..term {
        let temp = b;
        b += a;
        a = temp;
    }
    black_box(b)
}

fn keep_cpu_busy(duration: Duration) {
    let start = Instant::now();
    let mut counter = 0u64;
    while start.elapsed() < duration {
        // Prevent compiler from optimizing away the loop
        counter = black_box(counter.wrapping_add(1));
    }
}

struct LargeDrop {
    drop_duration: Duration,
}

impl LargeDrop {
    fn fake_compute(compute_duration: Duration, drop_duration: Duration) -> Self {
        keep_cpu_busy(compute_duration);
        Self { drop_duration }
    }
}

impl Drop for LargeDrop {
    fn drop(&mut self) {
        keep_cpu_busy(self.drop_duration);
    }
}

fn main() {
    let mut runner = Runner::setup_from_cmdline();

    runner.run("fibonacci_recursive_match", |bencher| {
        bencher.bench(|| {
            fibonacci_recursive_match(black_box(20));
        });
    });
    runner.run("fibonacci_recursive_if", |bencher| {
        bencher.bench(|| {
            fibonacci_recursive_if(black_box(20));
        });
    });
    runner.run("fibonacci_iterative", |bencher| {
        keep_cpu_busy(Duration::from_secs(5));
        bencher.bench(|| {
            fibonacci_iterative(black_box(20));
        });
    });
    runner.run("Vec::sort", |bencher| {
        keep_cpu_busy(Duration::from_secs(5));
        bencher.bench_with_inputs(
            || create_unsorted_vec(1000000),
            |mut vec| {
                vec.sort();
            },
        );
    });

    runner.run("LargeDrop", |bencher| {
        bencher.bench(|| LargeDrop::fake_compute(Duration::from_secs(1), Duration::from_secs(1)));
    });

    runner.run("LargeDrop2", |bencher| {
        bencher.bench_with_inputs(
            || LargeDrop::fake_compute(Duration::from_secs(0), Duration::from_secs(1)),
            |l| l,
        );
    });
}
