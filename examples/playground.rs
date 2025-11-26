use std::hint::black_box;

use quick_bench::{Bencher, GenericBencher, NoExtraMetric, Runner, cpu_time::CpuLoad};

fn fibonacci_recursive_if(n: i32) -> i32 {
    if n <= 0 {
        0
    } else if n == 1 {
        1
    } else {
        fibonacci_recursive_if(n - 1) + fibonacci_recursive_if(n - 2)
    }
}
fn work() {
    let mut d = 0;
    for n in 0..50_000 {
        for m in 0..50_000 {
            d = black_box(d * n * m);
        }
    }
}
fn main() {
    let mut runner = Runner::<CpuLoad>::new().setup_from_cmdline();

    runner.run_with(
        ("hello", |bencher: &mut GenericBencher<_>, _| {
            bencher.bench(|| fibonacci_recursive_if(10));
        }),
        [()],
    );

    runner.run_with(
        ("fibonnacci", |bencher: &mut GenericBencher<_>, n| {
            bencher.bench(|| fibonacci_recursive_if(n));
        }),
        [10, 20, 30, 32],
    );

    runner.run("lol", |bencher| {
        bencher.bench(|| {
            rayon::broadcast(|_| work());
        });
    });
}
